import torch
import pdb
import torch.nn.functional as F
import pdb
from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np

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