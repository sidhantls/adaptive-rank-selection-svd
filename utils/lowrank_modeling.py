import torch
from tqdm import tqdm
import pdb
from utils import train_utils
from utils import adaptive_rank_selection


def replace_with_lowrank_linear(model, args, svd_info={}):
    """
    Replace all linear layers in a PyTorch model with low-rank layer using SVD. This is used 
    before training

    Args:
        model (torch.nn.Module): The PyTorch model in which linear layers will be replaced.

    Returns:
        None
    """
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, torch.nn.Linear):
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

    i = 0
    for name, module in tqdm(model.named_modules(), total=total_len, desc='Replacing Linear with Low-Rank Layers', mininterval=5):
        if 'lm_head' in name:
            print('Ignored low-rank decomposition on logits layer')

        elif module in linear_info:
            info = linear_info[module]

            if svd_info: 
                svd_vector = svd_info[info['full_name']]
            else: 
                svd_vector = None

            if args.layer_type == 'simple':
                new_module = adaptive_rank_selection.LowrankLinearSimple(module, svd_vector, alpha=args.alpha, niter=2, tau=args.tau)
            elif args.layer_type == 'adaptive':
                new_module = adaptive_rank_selection.LowrankLinear(module, svd_vector, alpha=args.alpha, niter=2, tau=args.tau)
            elif args.layer_type == 'struct_pruning':
                new_module = adaptive_rank_selection.LowrankLinearStructPruning(module, svd_vector, alpha=args.alpha, niter=2, tau=args.tau)
            else:
                raise NotImplementedError(f"Unsupported layer_type {args.layer_type} in replace_linear_with_svd")

            setattr(info["father"], info["name"], new_module)

            del linear_info[module]
            del module
            torch.cuda.empty_cache()

            i += 1
            if i > 10 and args.debug:
                break

    torch.cuda.empty_cache()
    print('Replaced linear layers with low-rank layers.')

    
def get_compression_layers(model):
    """
    Returns the model parameters that controls the singular value selection, utilized for compression loss
    """
    compression_params = []
    for name, param in model.named_parameters():
        if 'E_train' in name:
            compression_params.append(param)

    return compression_params

@torch.no_grad()
def get_compression_metadata(model):
    """
    Returns a dataset containing compresison metadata. For eg, retrieves the 
    
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
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    compression_logs = [] 
    for module in linear_info: 
        layer_name = linear_info[module]['name']
        layer_idx = linear_info[module]['full_name'].split('.')[2]
        assert layer_idx.isnumeric(), f'layer_idx not numeric: {layer_idx}'

        mask = module.calculate_mask(is_training=False)
        r = int(round(mask.sum().item()))
        param_ratio = r * (module.in_features + module.out_features) / (module.in_features * module.out_features)

        compression_logs.append({'layer_idx': layer_idx, 'layer_name': layer_name, 'param_ratio': param_ratio, 'in_features': module.in_features, 'out_features': module.out_features, 
                                 'length': len(mask), 'topk': r, 'mask': mask.tolist()}
                                 )

    return compression_logs


class CompressionCalculator:
    """
    Aggregates lowrank layers for further use - compression calculation, loss etc 
    
    """
    def __init__(self, model, total_params):
        # count num parameters without the lowrank layers
        self.params1 = 0 
        for name, param in model.named_parameters():
            has_matching_params = any(item in name for item in ['UE', 'V_t', 'E_train'])
            if has_matching_params:
                pass 
            else:
                self.params1 += param.numel()
        
        self.lowrank_layers = [] 
        for _, module in model.named_modules():
            if 'Lowrank' in str(module)[:7]:
                self.lowrank_layers.append(module) 

        self.total_params = total_params

    
    def get_compression(self):
        params2 = 0 
        params_wo_lowrank = 0 
        for module in self.lowrank_layers: 
            if module.E_train_mask is None:
                with torch.no_grad():
                    rank = module.calculate_mask(is_training=False).sum().item()
            else:
                rank = module.E_train_mask.detach().sum().item()

            params_with_compression =  rank * (module.in_features + module.out_features)
            params_wo_compression = module.in_features * module.out_features

            # in reality, layer is not compressed when compression is huge
            if params_with_compression/params_wo_compression < 1.:
                params2 += params_with_compression
            else:
                params2 += params_wo_compression

            params_wo_lowrank +=  params_wo_compression

        compression = (self.params1 + params2) / (self.params1 + params_wo_lowrank)
        return compression
    
    def get_sv_ratio(self):
        keep_ratio = 0. 
        for module in self.lowrank_layers:
            keep_ratio += module.calculate_mask(is_training=True).mean().item()

        return keep_ratio/len(self.lowrank_layers)

    def get_compression_and_sv(self):
        params2 = 0
        params_wo_lowrank = 0
        sv_ratio = 0.
        for module in self.lowrank_layers:
            if module.E_train_mask is None:
                with torch.no_grad():
                    E_train_mask = module.calculate_mask(is_training=False).detach()
            else:
                E_train_mask = module.E_train_mask.detach()
            
            rank = (E_train_mask > 0.5).sum().item()
            sv_ratio += rank/len(E_train_mask)

            params_with_compression =  rank * (module.in_features + module.out_features)
            params_wo_compression = module.in_features * module.out_features

            # in reality, layer is not compressed when compression is huge
            if params_with_compression/params_wo_compression < 1.:
                params2 += params_with_compression
            else:
                params2 += params_wo_compression

            params_wo_lowrank +=  params_wo_compression

        compression = (self.params1 + params2) / (self.params1 + params_wo_lowrank)
        sv_ratio = sv_ratio/len(self.lowrank_layers)

        return compression, sv_ratio

