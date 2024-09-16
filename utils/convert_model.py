import torch
from tqdm import tqdm
import pdb
import numpy as np 
import pandas as pd 
from transformers import AutoModelForCausalLM
from collections import defaultdict


def get_mapping_dict(compression_stats):
    """
    Creates a dictionary that maps layer number and layer name to the number of singular values to retain. Saved and
    required when initializing a new llama model
    """

    mapping = defaultdict(dict)
    mapping_to_masks =  defaultdict(dict)
    for row in compression_stats:
        mapping[row['layer_idx']][row['layer_name']] = row['topk']
        mapping_to_masks[row['layer_idx']][row['layer_name']] = row['mask']

    return mapping, mapping_to_masks


def convert_linear_to_compressed(module):
    """
    Convert the low-rank linear model used in training to a compressed model, to be used during evalution 
    """

    E_train_mask = module.calculate_mask(is_training=False, return_topk=True)
    m,n,r =  module.UE.size(0), module.V_t.size(1), E_train_mask.sum().item()
    compression_ratio = (m*r + n*r)/(m*n)

    E_train_mask = E_train_mask.to(module.UE.device).bool() 

    # if compression is achieved, create lowrank layer 
    if compression_ratio < 1.:
        UE = module.UE[:, E_train_mask]
        V_t = module.V_t[E_train_mask, :]
        new_module = LinearLowRank(UE, V_t)
    
    else: 
        print(f'Compress not achieved for module ignore low-rank conversion:{module}')
        W_new = module.UE @ module.V_t
        new_module = torch.nn.Linear(W_new.shape[1], W_new.shape[0], bias=False)
        new_module.weight.data = W_new.contiguous()
        r = None

    return new_module, r

class LinearLowRank(torch.nn.Module):
    def __init__(self, UE, V_t, init_conig={}):
        super(LinearLowRank, self).__init__()
        """
        More efficient in the forward pass by avoiding first materialization of the weight

        Inputs: Linear layer to perform ASVD on.
        Approach: Parameter + gumbel sigmoid to generate mask
        """
        if not init_conig:
            self.in_features = int(V_t.shape[1])
            self.out_features = int(UE.shape[0])
            self.rank = int(V_t.shape[0])

            self.V_t = torch.nn.Linear(V_t.shape[1], V_t.shape[0], bias=False)
            self.UE = torch.nn.Linear(UE.shape[1], UE.shape[0], bias=False)
            self.V_t.weight.data = V_t.contiguous()
            self.UE.weight.data = UE.contiguous()
            print(f'Created UE: in={UE.shape[1]}, out={ UE.shape[0]}')
            print(f'Created V_t: in={V_t.shape[1]}, out={V_t.shape[0]}')
        else:
            self.in_features = init_conig['in_features']
            self.out_features = init_conig['out_features']
            self.rank = init_conig['rank']

            self.V_t = torch.nn.Linear(self.in_features, self.rank, bias=False)
            self.UE = torch.nn.Linear(self.rank, self.out_features, bias=False)

    def forward(self, inputs):
        x = self.V_t(inputs)
        return self.UE(x)

    def __str__(self):
        return f"LinearLowRank(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank})"

    def __repr__(self):
        return self.__str__()
    

def replace_with_compressed_layer(model):
    """
    Replace all the low-rank decomposed full-rank layers with layers that only contain the selected singular values.

    """
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if hasattr(raw_linear, 'E_train'):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    for total_len, _ in enumerate(model.named_modules()):
        pass
    
    replace_config = {} 
    i = 0
    for name, module in tqdm(model.named_modules(), total=total_len, desc='Replacing Linear with Low-Rank Layers', mininterval=5):
        if module in linear_info:
            info = linear_info[module]

            compressed_module, r = convert_linear_to_compressed(module)

            setattr(info["father"], info["name"], compressed_module)

            del linear_info[module]
            del module
            
            i += 1
            if i % 10 == 0:
                torch.cuda.empty_cache()

            # if no compression done, ignore adding to config
            if r == None:
                continue

            tokens = name.split('.')
            layer_idx, layer_name = int(tokens[2]), tokens[-1]

            if layer_idx not in replace_config: 
                replace_config[layer_idx] = {} 
            
            replace_config[layer_idx][layer_name] = r

    torch.cuda.empty_cache()
    print('Replaced low-rank layers with compressed low-rank layers.')
    return model, replace_config 
