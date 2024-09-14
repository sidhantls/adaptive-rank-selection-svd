from datasets import load_dataset, Dataset
import numpy as np
import random
import torch
import pdb
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import time
import os 
import pickle
import pdb 


def set_random_state(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def sample_documents_from_dataset_packing(dataset, args):
    """
    Samples a specified number of documents from a dataset for training and testing.

    Args:
        dataset (list[str]): A list of documents (strings) from which to sample.
        num_train_samples: The number of samples to include in the training set. Defaults to 128.
        num_test_samples: The number of samples to include in the test set. Defaults to 128.
        max_len: The maximum tokens of the sampled documents. Defaults to 512.

    Returns:
        train and testing documents, each of which are lit of strings
    """
    
    new_dataset = []
    left, right = 0, 1 
    document = dataset[left]
    while len(new_dataset) < (args.num_train_samples + args.num_test_samples) * 5 and right < len(dataset):
        if len(dataset[right].split()) < 100: # ignore less than 100 words, minimize the number of different documents 
            right += 1 
            continue
    
        document = document + ' ' + dataset[right]

        if len(document.split()) >= args.max_length:
            new_dataset.append(" ".join(document.split()))
            left = right + 1 
            right = left + 1
            document = dataset[left]
        else:
            right += 1 

    new_dataset = list(map(clean_text, new_dataset))
    train_docs, test_docs = train_test_split(new_dataset, train_size=args.num_train_samples, test_size=args.num_test_samples, random_state=args.seed)
    print(f'Number of train docs: {len(train_docs)}, number of test docs: {len(test_docs)}')

    return train_docs, test_docs


def get_dataloaders(
        tokenizer,
        args,
        dataset_name="wikitext2",
        random_state=42,
        ):
    """
    Function to generate PyTorch DataLoaders for training and testing a language model.

    Args:
        tokenizer (Tokenizer): The tokenizer object used to tokenize the text data.
        dataset_name (str): The name of the dataset. Currently supports 'wikitext2'.
        num_train_samples (int): Number of samples to use for training.
        num_test_samples (int): Number of samples to use for testing.
        batch_size (int): Batch size for DataLoader.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_dataloader (DataLoader): DataLoader for training data.
        test_dataloader (DataLoader): DataLoader for testing data.
    """
    set_random_state(random_state)
    NUM_CALIB=256
    MAX_LEN_CALIB = 2048
    if args.debug:
        NUM_CALIB=2
        MAX_LEN_CALIB=10

    start = time.time()
    
    if dataset_name == "wikitext2":
        hf_dataset = load_wikitext(args)
    else:
        raise NotImplementedError('dataset_name not in available list in get_train_dl')
    print(f'Time taken to load datasets {time.time()-start: 0.2f}')

    # use pre-training dataset style with packing
    train_docs, test_docs = sample_documents_from_dataset_packing(hf_dataset['text'], args), args)

    if args.debug: 
        args.max_length = 10
        train_docs, test_docs = train_docs[:10], test_docs[:10]
        
    train_dataset = tokenizer(train_docs, max_length=args.max_length, truncation=True)
    test_dataset = tokenizer(test_docs, max_length=args.max_length, truncation=True)

    # calib_dataset = tokenizer(train_docs[:NUM_CALIB], max_length=args.max_length, truncation=True)
    calib_docs = get_calib_data(args, NUM_CALIB)
    calib_dataset = tokenizer(calib_docs, max_length=MAX_LEN_CALIB, truncation=True)

    train_dataset = Dataset.from_dict(train_dataset)
    test_dataset = Dataset.from_dict(test_dataset)
    calib_dataset = Dataset.from_dict(calib_dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create PyTorch DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=False)
    calib_loader = DataLoader(calib_dataset, batch_size=1, collate_fn=data_collator, shuffle=False)

    return train_dataloader, test_dataloader, calib_loader

def get_calib_data(args, nsamples, seqlen=2048, seed=3):
    """
    Returns calibration list of documents to use with ASVD/FWSVD
    From: https://github.com/hahnyuan/ASVD4LLM/blob/main/datautils.py
    """

    print(f" get_ptq_calib_data wikitext, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    
    load_path = os.path.join(args.cache_dir, 'wikitext-2-raw-v1.pkl')
    if not os.path.exists(load_path):
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        with open(load_path, 'wb') as f:
            pickle.dump(traindata, f)
    else:
        with open(load_path, 'rb') as f:
            traindata = pickle.load(f)
        
    tot_text = "\n\n".join(traindata["text"])

    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        traindataset.append(tot_text[i:j])
    return traindataset

def load_wikitext(args):
    hf_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    
    print('Num samples in raw dataset', len(hf_dataset))
    return hf_dataset

def clean_text(x):
    """
    Clean wikitext. Noticed in Wikitext these irrelevant strings are commonly present
    """
    return x.replace("@-@", " ").replace("@.@", " ").replace('= =', ' ')

if __name__ == '__main__':
    from transformers import AutoTokenizer

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataloader, test_dataloader = get_dataloaders(
        tokenizer, dataset_name="wikitext2", num_train_samples=256, num_test_samples=256)
