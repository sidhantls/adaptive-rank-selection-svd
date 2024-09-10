import torch 
from torch import nn 
from tqdm import tqdm 
import os 

@torch.no_grad()
def calib_input_distribution(model, calib_loader, method, args):
    """
    Modified from https://github.com/hahnyuan/ASVD4LLM/blob/main/modules/svd_linear.py
    """

    cache_path = os.path.join(args.cache_dir, 'asvd_cache.pth')
    if args.load_act_cache == True:
        return torch.load(cache_path)
    
    model.eval()

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean.detach().cpu()
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            ).detach().cpu()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.scaling_diag_matrix = 0
            module.register_forward_hook(hook)

    # get activation distribution
    for batch in tqdm(calib_loader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    # remove and save scaling_diag_matrix
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_scaling_diag_matrix[name] = module.scaling_diag_matrix.detach().cpu()

    torch.save(all_scaling_diag_matrix, cache_path)

    return all_scaling_diag_matrix

def calib_fisher_info(model, calib_loader, args):
    """
    Modified from https://github.com/hahnyuan/ASVD4LLM/blob/main/modules/svd_linear.py
    """
    cache_path = os.path.join(args.cache_dir, 'fischer_cache.pth')
    if args.load_act_cache == True:
        return torch.load(cache_path)

    model = model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = 0

    # get fisher info
    for batch_idx, batch in enumerate(tqdm(calib_loader)):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        out = model(input_ids=input_ids, labels=labels)        
        out[0].backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info += module.weight.grad.detach().pow(2).mean(0).detach().cpu()
        model.zero_grad()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt().detach().cpu()

    # remove and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_fisher_info[name] = module.fisher_info.detach().cpu()

    torch.save(all_fisher_info, cache_path)
    return all_fisher_info

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from data_utils import get_dataloaders
    import pdb

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    train_dataloader, test_dataloader, calib_loader = get_dataloaders(
        tokenizer, dataset_name="wikitext2", num_train_samples=9, num_test_samples=256, max_length=256, batch_size=4)
    
    method = "abs_mean"
    all_scaling_diag_matrix = calib_input_distribution(model, calib_loader, method)
    # pdb.set_trace()

    all_fisher_info = calib_fisher_info(model, train_dataloader)
    pdb.set_trace()
    

    
