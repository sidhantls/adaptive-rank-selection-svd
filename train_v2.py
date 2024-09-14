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

parser.add_argument("--epochs", type=int, default=5, help="The number of epochs")

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

parser.add_argument("--init_frac", type=float, default=1.0, help="Starting fraction of singular values to keep. Usued with gumbel sigmoid")

parser.add_argument("--act_aware", type=str, default='', help='Loss/activation aware SVD', choices=['', 'fisher', 'activation'])

parser.add_argument("--alpha", type=float, default=1., help="Alpha hyperparameter for act_aware")

parser.add_argument("--target_param_ratio", type=float, default=-1, help="If compression value is less than this, compression loss is scaled to 0")

parser.add_argument('--fix_length', action='store_true', default=False, help='Keep a fixed sequence length')

parser.add_argument('--save_model', type=str, default='reconstruct',  help='Method to save model', choices=['reconstruct', 'use_mask'])

parser.add_argument('--load_act_cache', action='store_true', default=False, help='Loads activation cache')

parser.add_argument('--load_act_path', type=str, default="", help='Loads activation cache from a particular directory')

parser.add_argument( "--lr_schedule", type=str, default="", choices=["", "plateau"], help="Type of learning rate scheduler to use.")

parser.add_argument( "--use_logits_loss", action='store_true', default=False, help="If this is true, it starts at acompression rate = 1. Currently, starting compression is more than one since we use full rank")

parser.add_argument("--seed", type=int, default=233, help="Seed used in experiment")

args = parser.parse_args()

args.tau=0.4
args.fix_compress_ratio = None
args.mask_eval_type = None 
args.ignore_first_layer = None 
args.scale_singular_values = None 
args.schedule_distillation = "" 
args.only_compress = "" 
args.layer_type = 'adaptive'
args.compress_loss = None 
args.distill_mode = None 

os.makedirs(args.cache_dir, exist_ok=True)

np.random.seed(args.seed)  # Set the seed for NumPy
torch.manual_seed(args.seed)  # Set the seed for PyTorch on CPU
torch.cuda.manual_seed_all(args.seed)  # Set the seed for PyTorch on all GPUs
torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CuDNN

if args.debug: 
    os.environ["WANDB_MODE"] = "offline"

wandb_writer = wandb.init(project="learn-to-compress-lrd3", name=args.exp_name, config=vars(args))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# load model 
if 'Llama-2' in args.model_name:
    tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    #tokenizer.pad_token_id=0
    print('Loaded llama tokenizer')
elif 'Llama-3' in args.model_name:
    tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    print('Loaded llama 3 tokenizer')
elif 'gemma' in args.model_name.lower():
    tokenizer = GemmaTokenizerFast.from_pretrained(args.model_name, cache_dir=args.cache_dir)
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
print('\nModel\n', model, '\n')
num_params_old = train_utils.count_parameters(model)
print(f'Model loaded in {time.time()-start: 0.2f} seconds')
print(f'Model dtype: {model.dtype}')

if torch.cuda.is_available():
    model = model.cuda()

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

optimizer = AdamW(model.parameters(), lr=args.lr)

args.fix_compress_ratio=None 

if args.fix_compress_ratio or args.compress_loss == 'mse': 
    with torch.no_grad():
        args.target_keep_ratio = compression_calculator.get_sv_ratio()
else:
    args.target_keep_ratio = None

lr_scheduler = None 
if args.lr_schedule and args.lr_schedule != 'plateau': 
    if args.lr_schedule == 'linear':
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=args.epochs * len(train_dl))

    elif args.lr_schedule == 'cycle': 
        lr_scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-3,
        max_lr=1e-2,
        step_size_up=args.epochs * len(train_dl) // 5,  # Number of steps to go from base_lr to max_lr
        mode='triangular',
        cycle_momentum=False
    )
    else:
        raise NotImplementedError('Unsupported Scheduler')
        
# Training loop
global_step, max_steps = 0, args.epochs * len(train_dl)
eval_interval = args.epochs // args.eval_freq

# evaluating before first epoch 
args.scale_distill = None

if not args.debug:
    harness_metrics = eval_utils.evaluate_with_harness(model, tokenizer, device=model.device, debug=args.debug, batch_size=args.batch_size)
    wandb.log({ **harness_metrics, 'step': global_step})
    print('first eval\n', harness_metrics)

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
        args.scale_distill = train_utils.schedule_distill_scale(global_step, max_steps, args.schedule_distillation, args.scale_distill)

        if args.distill_mode:
            distill_data_path = os.path.join(args.cache_dir, f"distill_cache/train_{batch_idx}.pt")
            distill_batch = torch.load(distill_data_path)
        else:
            distill_batch = {}

        with torch.autocast(device_type=model.device.type, dtype=train_precision, enabled=use_amp):
            loss, logits_loss, distill_loss, distill_kl, hs_loss, compression_loss, perplexity, keep_ratio, tv_loss, compression_ratio = train_utils.training_step(
                model, batch, batch_idx, compression_params, 
                tokenizer.pad_token_id, distill_batch, args, compression_calculator)

        scaler.scale(loss).backward()
        
        #scaler.unscale_(optimizer) # gradient clipping on unscaled gradients
        #clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if args.lr_schedule and 'plateau' not in args.lr_schedule:
            lr_scheduler.step()

        # Combining all metrics into one dictionary and logging with wandb in one line
        metrics = {
            "train/loss": loss.item(),
            "train/logits_loss": logits_loss,
            "train/compression_loss": compression_loss.item(),
            "train/distill_loss": distill_loss,
            "train/distill_logits_loss": distill_kl, 
            "train/distill_loss_hs": hs_loss,
            "train/target_keep_ratio": args.target_keep_ratio,
            "train/sv_keep_ratio": keep_ratio,
            "train/perplexity": perplexity.item(), 
            "train/scale_distill": args.scale_distill, 
            "train/lr": optimizer.param_groups[0]['lr'],
            "train/tv_loss": tv_loss,
            "train/compression_ratio": compression_ratio,
            "step": global_step
        }
        
        param_ratios.append(compression_ratio)

        if batch_idx % 2 == 0:
            wandb.log(metrics)

        global_step += 1
        epoch_loss += loss.item()
        num_batches += 1
        
        del distill_batch

        # eval every steps: for pre-training objective
        if batch_idx and args.eval_freq_steps and (batch_idx % args.eval_freq_steps) == 0:
            model = model.eval()
            metrics = train_utils.eval_model(model, test_dl, compression_params, tokenizer.pad_token_id, args, compression_calculator)
            harness_metrics = eval_utils.evaluate_with_harness(model, tokenizer, device=model.device, debug=args.debug, batch_size=args.batch_size)
            model = model.train()
            wandb.log({**metrics, **harness_metrics, 'step': global_step})

        window_size=250 # continue training for 500 more steps after target is reached
        current_mean = np.mean(param_ratios[-window_size:])
        short_mean = np.mean(param_ratios[-30:])
        
        # saturate learning rate 
        if args.lr_schedule == 'plateau' and len(param_ratios) > 30 and (short_mean-args.target_param_ratio) < 0.005:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr/2
            args.lr_schedule = 'plateau_saturated' # update string so learning rate is not edited again

            print('Compression level reached, saturating learning rate')

        is_compression_reached = len(param_ratios) > window_size and (current_mean - args.target_param_ratio) < 0.0015
 
        # if pre-training mode, early stop after 5% more steps if performance is constant
        if is_compression_reached: 
            print(f'\n\nCompression Ratio: {current_mean} reached for 500 steps, early stopping training...\n\n')
            break
    
    if is_compression_reached:
        break 

    if args.end_tau is not None and args.end_tau != args.start_tau: 
        old_tau = args.tau
        args.tau = lowrank_modeling.update_lowrank_tau(model, args.epochs, args.end_tau, start_tau=args.start_tau)
        print('Updated tau from {old_tau} to {args.tau}')
    
    if epoch and (epoch % args.eval_freq) == 0:
        model = model.eval()
        metrics = train_utils.eval_model(model, test_dl, compression_params, tokenizer.pad_token_id, args, compression_calculator)
        harness_metrics = eval_utils.evaluate_with_harness(model, tokenizer, device=model.device, debug=args.debug, batch_size=args.batch_size)
        model = model.train()
    
        wandb.log({**metrics, **harness_metrics, 'train/tau': args.tau, 'step': global_step})

    epoch_loss = epoch_loss/num_batches
    wandb.log({'train/epoch_loss': epoch_loss, 'step': epoch})

if not args.debug and not (epoch % args.eval_freq == 0) or is_compression_reached: # if is_compression_reached=True, then training was terminated
    metrics = train_utils.eval_model(model, test_dl, compression_params, tokenizer.pad_token_id, args, compression_calculator)
    harness_metrics = eval_utils.evaluate_with_harness(model, tokenizer, device=model.device, debug=False, batch_size=args.batch_size)
    wandb.log({**metrics, **harness_metrics, 'step': global_step})

print('Training complete.')

# log compression metadata: learnt weights per layer
compression_metadata = lowrank_modeling.get_compression_metadata(model)
stats_path = os.path.join(wandb_writer.dir, 'compression_stats.json')
with open(stats_path, 'w') as f:
    json.dump(compression_metadata, f)
wandb.Artifact(name="compression_metadata", type="dataset").add_file(stats_path)
   
if args.save_model:
    model = model.cpu()
    model_path = os.path.join(args.cache_dir, f'{wandb.run.id}_saved_model')
    os.makedirs(model_path, exist_ok=True)

    # save mapping 
    compression_map, compression_map_mask  = convert_model.get_mapping_dict(compression_metadata)
    with open(os.path.join(model_path, 'compression_map.json'), 'w') as f:
        json.dump(compression_map, f)
    wandb.Artifact(name="compression_map", type="dataset").add_file(os.path.join(model_path, 'compression_map.json'))
    
    if args.save_model == 'reconstruct': 
        model, lowrank_config = convert_model.replace_with_compressed_layer(model)
    elif args.save_model == 'use_mask': 
        del optimizer; del model; del compression_params; torch.cuda.empty_cache()
        model, lowrank_config = convert_model.replace_with_compressed_layer_13b(args, svd_info, compression_map_mask)   
    else:
        raise NotImplementedError('arg.save_model {args.save_model} not recognised')     

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
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    harness_metrics_full = eval_utils.evaluate_with_harness_full(model, tokenizer, device, debug=args.debug, batch_size=args.eval_batch_size)
    harness_metrics_full = {'final_' + k: v for k, v in harness_metrics_full.items()}
    wandb.log({**harness_metrics_full, 'step': global_step})
    print('Final harness results: \n', harness_metrics_full, '\n')
