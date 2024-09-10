import argparse
import json
import os
import pickle
import time

import pdb
import torch
import wandb
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

from transformers import (
    AdamW,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizerFast
)

from utils import (
    train_utils,
    loss_aware,
    convert_model,
    eval_utils,
    lowrank_modeling
)
from utils.data_utils import get_dataloaders

import numpy as np
from utils import adaptive_rank_selection

parser = argparse.ArgumentParser(description="Transformer model training and evaluation")

parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="The name or path of the pre-trained model to use.")

parser.add_argument("--batch_size", type=int, default=2,
                    help="Batch size for model")

parser.add_argument("--eval_batch_size", type=int, default=14,
                    help="Batch size for model")

parser.add_argument("--num_train_samples", type=int, default=256,
                    help="The number of samples to use for the training dataset.")

parser.add_argument("--num_test_samples", type=int, default=256,
                    help="The number of samples to use for the test dataset.")

parser.add_argument("--max_length", type=int, default=512, help="Maximum number of input tokens")

parser.add_argument("--lr", type=float, default=1e-5,
                    help="Learning rate")

parser.add_argument("--eval_freq", type=int, default=1,
                    help="Evaluate after n epochs. If 1, evaluation after every epoch")

parser.add_argument("--eval_freq_steps", type=int, default=0,
                    help="Default off. Integer number of the number of steps after which to run evaluation in an epoch")

parser.add_argument('--debug', action='store_true', default=False, help='Debug mode, faster execution')

parser.add_argument("--exp_name", type=str, default='test', help="Experiment name")

parser.add_argument("--cache_dir", type=str, default='train_cache/', help='Directory where distillation cache is stored')

parser.add_argument('--eval_full', action='store_true', default=False, help='Run evaluation on large dataset when training is complete')

parser.add_argument("--act_aware", type=str, default='', help='Loss/activation aware SVD', choices=['', 'fisher', 'activation'])

parser.add_argument("--alpha", type=float, default=1., help="Alpha hyperparameter for act_aware")

parser.add_argument("--target_param_ratio", type=float, help="Target compression", required=True)

parser.add_argument('--save_model', type=str, default='reconstruct',  help='Method to save model', choices=['reconstruct', 'use_mask'])

parser.add_argument('--load_act_cache', action='store_true', default=False, help='Loads activation cache')

parser.add_argument('--load_act_path', type=str, default="", help='Loads activation cache from a particular directory')

parser.add_argument("--seed", type=int, default=233, help="Seed used in experiment")

parser.add_argument("--tau", type=float, default=0.4, help="Tau for gumbel sigmoid")

parser.add_argument("--lambda_scale", type=float, default=16., help="Scale factor for compression regularization loss")

parser.add_argument("--gamma_scale", type=float, default=10., help="Scale factor of alignment loss")

args = parser.parse_args()

# constant
args.layer_type='adaptive'
args.distill_mode = None
args.epochs=1

os.makedirs(args.cache_dir, exist_ok=True)

np.random.seed(args.seed)  # Set the seed for NumPy
torch.manual_seed(args.seed)  # Set the seed for PyTorch on CPU
torch.cuda.manual_seed_all(args.seed)  # Set the seed for PyTorch on all GPUs
torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CuDNN

if args.debug: 
    os.environ["WANDB_MODE"] = "offline"

wandb_writer = wandb.init(project="learn-to-compress-lrd2", name=args.exp_name, config=vars(args))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# load model 
if 'Llama-2' in args.model_name:
    tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    print('Loaded llama tokenizer')
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

train_dl, test_dl, calib_loader = get_dataloaders(tokenizer,
                                    args,
                                    dataset_name="wikitext2",
                                    )

if torch.cuda.is_available():   
    torch_dtype, use_amp = torch.float32, True
    train_precision = torch.bfloat16 # mixed precision 
else:
    torch_dtype, use_amp = torch.float32, False
    train_precision = torch.float16

start = time.time()
model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
num_params_old = train_utils.count_parameters(model)
print(f'Model loaded in {time.time()-start: 0.2f} seconds')
print(f'Model dtype: {model.dtype}')

if torch.cuda.is_available():
    model = model.cuda()
    
print('Memory available after generating distillation dataset') 
train_utils.print_nvidia_smi()

print(f'\nModel pushed to {device} after distillation', model.device)

svd_info = {} 
if args.act_aware: 
    print(f'\nGenerating loss/activation aware SVD dataset: {args.act_aware}\n')

    if args.load_act_path:
        assert os.path.exists(args.load_act_path), f"File not found: {args.load_act_path}"
        svd_info = torch.load(args.load_act_path)
    elif args.act_aware == 'fisher':
        svd_info = loss_aware.calib_fisher_info(model, calib_loader, args=args)
    elif args.act_aware == 'activation':
        svd_info = loss_aware.calib_input_distribution(model, calib_loader, method='abs_mean', args=args)
    else:
        raise NotImplementedError(f'Activation aware {args.act_aware} not supported')

# add low-rank decomposed layers and set grads 
model = model.cpu(); torch.cuda.empty_cache() # move to cpu for layer editing
lowrank_modeling.replace_with_lowrank_linear(model, args, svd_info)
train_utils.configure_required_grad(model)
model = model.to(device)

# pass in uncompressed model 
compression_calculator = lowrank_modeling.CompressionCalculator(model, total_params=num_params_old*1e9)
start = time.time()
current_compression = compression_calculator.get_compression()
print('Time taken to get current compression rate (seconds):', time.time()-start)

train_utils.print_nvidia_smi()

# singular value selection parameters, required for loss
compression_params = lowrank_modeling.get_compression_layers(model)

optimizer = Adam(model.parameters(), lr=args.lr)
#optimizer = Adam(model.parameters(), lr=args.lr)

# Training loop
global_step, max_steps = 0, args.epochs * len(train_dl)
eval_interval = args.epochs // args.eval_freq
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# variable that stores number of epochs for which compressionrate has been reached. 
reached_compression_steps = 0
param_ratios = []
is_compression_reached = False 

print('Starting training..')
for epoch in range(args.epochs):
    model = model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(train_dl, desc=f"Train Epoch {epoch+1}", mininterval=5)):
        # if args.distill_mode:
        #     distill_data_path = os.path.join(args.cache_dir, f"distill_cache/train_{batch_idx}.pt")
        #     distill_batch = torch.load(distill_data_path)
        # else:
        #     distill_batch = {}

        with torch.autocast(device_type=model.device.type, dtype=train_precision, enabled=use_amp):
            loss, logits_loss, r_align_loss, r_loss, perplexity, keep_ratio, current_param_ratio, lambda_scale = adaptive_rank_selection.training_step(model, batch, tokenizer.pad_token_id, args, compression_calculator)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Combining all metrics into one dictionary and logging with wandb in one line
        metrics = {
            "train/loss": loss.item(),
            "train/logits_loss": logits_loss,
            "train/r_align_loss": r_align_loss.item(),
            "train/sv_keep_ratio": keep_ratio,
            "train/perplexity": perplexity.item(), 
            "train/r_loss": r_loss, 
            "train/lr": optimizer.param_groups[0]['lr'],
            "train/target_param_ratio": args.target_param_ratio,
            "train/compression_ratio": current_param_ratio,
            "train/lambda_scale": lambda_scale,
            "step": global_step
        }
        
        param_ratios.append(current_param_ratio)

        if batch_idx % 2 == 0:
            wandb.log(metrics)

        global_step += 1
        epoch_loss += loss.item()
        num_batches += 1
        
        # eval every steps: for pre-training objective
        if batch_idx and args.eval_freq_steps and (batch_idx % args.eval_freq_steps) == 0:
            model = model.eval()
            # adaptive_rank_selection.freeze_model_masks(model, should_freeze=True)

            metrics = adaptive_rank_selection.eval_model(model, test_dl, tokenizer.pad_token_id, args, compression_calculator)
            harness_metrics = eval_utils.evaluate_with_harness(model, tokenizer, device=model.device, debug=args.debug, batch_size=args.batch_size)
        
            wandb.log({**metrics, **harness_metrics, 'step': global_step})
            # adaptive_rank_selection.freeze_model_masks(model, should_freeze=False)
            model = model.train()

        window_size = 5 # continue training for X more steps after target is reached
        current_mean = np.mean(param_ratios[-window_size:])

        is_compression_reached = len(param_ratios) > window_size and (current_mean - args.target_param_ratio) < 0.0015
 
        # if pre-training mode, early stop after 5% more steps if performance is constant
        if is_compression_reached: 
            print(f'\nCompression Ratio: {current_mean} reached for {window_size} steps, early stopping training...\n')
            break

    if is_compression_reached:
        break 
    
    epoch_loss = epoch_loss/num_batches
    wandb.log({'train/epoch_loss': epoch_loss, 'step': epoch})

# if not args.debug and not (epoch % args.eval_freq == 0) or is_compression_reached: # if is_compression_reached=True, then training was terminated
#     model = model.eval(); adaptive_rank_selection.freeze_model_masks(model, should_freeze=True)

#     metrics = adaptive_rank_selection.eval_model(model, test_dl, tokenizer.pad_token_id, args, compression_calculator)
#     harness_metrics = eval_utils.evaluate_with_harness(model, tokenizer, device=model.device, debug=args.debug, batch_size=args.batch_size)

#     wandb.log({**metrics, **harness_metrics, 'step': global_step})
#     adaptive_rank_selection.freeze_model_masks(model, should_freeze=False)
#     model = model.train()

print('Training complete.')

# log compression metadata: learnt weights per layer
compression_metadata = lowrank_modeling.get_compression_metadata(model)
stats_path = os.path.join(wandb_writer.dir, 'compression_stats.json')
with open(stats_path, 'w') as f:
    json.dump(compression_metadata, f)
wandb.Artifact(name="compression_metadata", type="dataset").add_file(stats_path)
   
# evaluate the final model as well
if args.eval_full:
    if torch.cuda.is_available(): model = model.cuda()
    model = model.eval()
    model = model.half()
    # adaptive_rank_selection.freeze_model_masks(model, should_freeze=True)

    harness_metrics_full = eval_utils.evaluate_with_harness_full(model, tokenizer, device, debug=args.debug, batch_size=args.eval_batch_size)
    harness_metrics_full = {'final_bc_' + k: v for k, v in harness_metrics_full.items()} # evaluate before converting the model, sanity check
    wandb.log({**harness_metrics_full, 'step': global_step})
    print('Final harness results: \n', harness_metrics_full, '\n')

    model = model.float() # back to fp32, for model convertion. may not even matter

if args.save_model:
    model = model.cpu()
    model_path = os.path.join(args.cache_dir, f'{wandb.run.id}_saved_model')
    os.makedirs(model_path, exist_ok=True)

    # save mapping 
    compression_map, compression_map_mask  = convert_model.get_mapping_dict(compression_metadata)
    with open(os.path.join(model_path, 'compression_map.json'), 'w') as f:
        json.dump(compression_map, f)
    wandb.Artifact(name="compression_map", type="dataset").add_file(os.path.join(model_path, 'compression_map.json'))
    
    model, lowrank_config = convert_model.replace_with_compressed_layer(model)

    num_params_new = train_utils.count_parameters(model)
    compression_stats = { "compression_stats/new_params_billion": num_params_new, "compression_stats/old_params_billion": num_params_old, "compression_stats/compression_ratio": num_params_new / num_params_old }
    print(f"\n\n--Compression Stats---\n{json.dumps(compression_stats, indent=4)}")
    wandb.log({**compression_stats, 'step': global_step})

    with open(os.path.join(model_path, 'lowrank_config.json'), 'w') as f:
        json.dump(lowrank_config, f)
    wandb.Artifact(name="lowrank_config", type="dataset").add_file(os.path.join(model_path, 'lowrank_config.json'))

    model = model.half()
    model.save_pretrained(model_path)
    print(f'Model save: {model_path}')

# evaluate the final model as well
if args.eval_full:
    if torch.cuda.is_available(): model = model.cuda()
    model = model.eval(); 
    # adaptive_rank_selection.freeze_model_masks(model, should_freeze=True)

    harness_metrics_full = eval_utils.evaluate_with_harness_full(model, tokenizer, device, debug=args.debug, batch_size=args.eval_batch_size)
    harness_metrics_full = {'final_' + k: v for k, v in harness_metrics_full.items()}
    wandb.log({**harness_metrics_full, 'step': global_step})
    print('Final harness results: \n', harness_metrics_full, '\n')
