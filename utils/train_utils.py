import torch.nn.functional as F
from tqdm import tqdm
import torch

def configure_required_grad(model):
    """
    Set which layers requires gradients and which doesn't.

    None of the layers require grad except for trainable singular values
    """
    non_trainable = trainable = 0
    for name, param in model.named_parameters():
        if 'E_train' in name:
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False
            non_trainable += 1

    print(
        f'Layers that require gradients configure. Number of trainable layers: {trainable}, fraction: {trainable/(trainable+non_trainable): 0.2f}')

def print_nvidia_smi():
    import subprocess
    
    try:
        # Run nvidia-smi command
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', check=True)
        
        print(result.stdout)
    
    except:
        print(f"Error: no nvidia-smi to check gpu util")
    
def count_parameters(model):
    """
    Calculate the number of parameters in a model and return the count in billions.
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_billion = total_params / 1e9
    return total_params_in_billion

def push_to_multi_gpu(model):
    """
    Pushes MLP layers with numbers between 3 and 20 in their names to one GPU (device 1),
    and the rest of the layers to another GPU (device 0).
    """

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
    else:
        raise RuntimeError("At least two CUDA devices are required for this operation.")

    # Push modules to the corresponding devices
    for name, module in model.named_modules():
        # Check if 'mlp' is in the name and if any number between 3 and 20 is present
        if  'gate_proj' in name or 'up_proj' in name:
            module.to(device1)  # Push to device 1
            print(f"{name} pushed to device 1 (cuda:1)")
        else:
            module.to(device0)  # Push to device 0
            print(f"{name} pushed to device 0 (cuda:0)")

    return model