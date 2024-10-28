"""
Contains implementation regarding https://aclanthology.org/2024.naacl-long.13.pdf
"""
import torch 
from torch import nn
import torch.nn.functional as F
import pdb
from collections import defaultdict
from tqdm import tqdm 

def gumbel_sigmoid(logits, tau=0.5):
    """Apply Gumbel Sigmoid to logits"""

    def sample_gumbel(shape, dtype, device, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape, device=device, dtype=dtype)
        return -torch.log(-torch.log(U + eps) + eps)

    gumbel_noise = sample_gumbel(logits.shape, logits.dtype, logits.device)
    gumbel_logits = logits + gumbel_noise
    y_soft = torch.sigmoid(gumbel_logits / tau)
    return y_soft

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward pass: apply threshold (> 0.5)
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass: STE returns the gradient of the original input
        return grad_output

class Hypernet_GRU(nn.Module):
    """
    Part 1 of the hypernet, the GRU network, which is a global layer: one for the entire network 
    
    """
    def __init__(self, num_layers, input_size, hidden_size):
        super(Hypernet_GRU, self).__init__()
        
        self.bi_gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.activation = nn.GELU()
        #self.z = torch.randn(1, num_singular_values, input_size)  # Normal distribution input
        self.z = torch.nn.Parameter(torch.randn(1, num_layers, input_size)) # use nn.param for device and type consistency
        self.z.requires_grad=False

    def forward(self):
        """
        Input: (batch_size, timesteps, input_size)
        Output: (batch_size, timesteps, output_size)
        """
        self.z.requires_grad=False 
        out, _ = self.bi_gru(self.z)
        out = self.layer_norm(out)
        out = self.activation(out)[0, :, :]
        return out
    
class LowrankLinear(torch.nn.Module):
    def __init__(self, current_layer, svd_vector, alpha=1., niter=2, tau=0.4):
        """
        Decomposes the weight in a linear layer into its singular vectors and values and introduces a learnable mask.

        Args:
            current_layer (nn.Linear): Current linear layer to be decomposed.
            svd_vector (torch.Tensor, optional): Weight scales for weighted SVD.
            alpha (float): Hyperparameter for weighted ASVD (default: 1.0).
            niter (int): Number of SVD iterations (default: 2).
            tau (float): Temperature of Gumbel sigmoid (lower values create harder boundaries) (default: 0.1).
        """
        super(LowrankLinear, self).__init__()

        if not isinstance(current_layer, torch.nn.Linear):
            raise ValueError(f"Expected input into SVDLayer be of instance nn.Linear, got {type(current_layer)}")
        
        # bias to add to gumbel sigmoid to ensure full rank is selected
        self.b = 3.

        dtype = current_layer.weight.dtype
        self.in_features, self.out_features = current_layer.in_features, current_layer.out_features
        self.rank = min(current_layer.weight.shape[1], current_layer.weight.shape[0])

        weight = current_layer.weight.float()
        layer_device = weight.device

        if torch.cuda.is_available():
            weight = weight.cuda()

        if svd_vector is not None: 
            svd_vector += 1e-6 # division by zero
            svd_vector = svd_vector.to(weight.device)**alpha
            weight = weight * svd_vector.unsqueeze(0)
        
        with torch.no_grad():
            U, E, V = torch.svd_lowrank(weight, 
                                        q=self.rank,
                                        niter=niter)
        
        if svd_vector is not None: 
            V = V / svd_vector.unsqueeze(1)
        
        U, E, V = U.to(layer_device), E.to(layer_device), V.to(layer_device)

        assert len(E.shape) == 1, 'expected singular values to have only one dim'

        # precompute EV for efficency
        self.UE = torch.nn.Parameter((U * E.unsqueeze(0)).to(dtype), requires_grad=True)
        self.V_t = torch.nn.Parameter(V.T.to(dtype), requires_grad=True)
        self.E = E
        self.E.requires_grad=False

        # 128 is hidden_dim of Bi-GRU
        self.E_train = nn.Linear(128, len(self.E)) 
        self.tau = tau
        self.global_hypernet_state = None
        self.E_train_mask = None

    def forward(self, inputs):
        """
        Computes forward pass with selection of singular values through predicted mask.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        #if self.use_stored_masks: # doesnt work for some reason, ignore
        #    return self.E_train_mask > 0.5
        
        E_train_mask = self.calculate_mask(is_training=self.training)
        if self.training:
            self.E_train_mask = E_train_mask 
        inputs = inputs.transpose(1, 2)

        if inputs.device != self.V_t.device: # multi-gpu setup
            inputs = inputs.to(self.V_t.device)
        if E_train_mask != self.V_t.device:
            E_train_mask = E_train_mask.to(self.V_t.device)
        output = (self.UE * E_train_mask.unsqueeze(0)) @ (self.V_t @ inputs)
        output = output.transpose(1, 2)

        return output
    
    def calculate_mask(self, is_training, return_topk=False):
        """
        Calculates the mask for singular value selection. During training, it uses Gumbel-Sigmoid approximation.
        During evaluation, various non-differentiable operations can be used

        Args:
            is_training (bool): Whether the model is in training mode.
            return_probs (bool): Whether to return probabilities instead of binary mask (default: False).

        Returns:
            torch.Tensor: Mask for singular value selection.
        """
        if isinstance(self.global_hypernet_state, type(None)):
            raise TypeError("Expected self.global_hypernet_state to be of type tensor in LowrankLinear")
        
        if is_training or self.E_train_mask is None:
            logit_mask = self.E_train(self.global_hypernet_state.to(self.E_train.weight.device) + self.b
            E_train_mask = gumbel_sigmoid(logit_mask, tau=self.tau)
            E_train_mask = STE.apply(E_train_mask)
            self.E_train_mask = E_train_mask
        else:
            E_train_mask = self.E_train_mask

        if not is_training:
            E_train_mask = E_train_mask > 0.5
            topk_mask = torch.zeros_like(E_train_mask, device=E_train_mask.device, requires_grad=False)
            topk_mask[:E_train_mask.sum().item()] = 1.
            return topk_mask

        return E_train_mask

    def __str__(self):
        return f"LowrankLinear(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank})"

    def __repr__(self):
        return self.__str__()
    
#def calculate_r_align(compression_calculator):
#    """
#    Loss to align learned mask to singular value properties
#    """
#    loss = 0.
#    for module in compression_calculator.lowrank_layers: 
#        with torch.no_grad():
#            k = module.calculate_mask(False).sum().item()
#            m = torch.zeros_like(module.E_train_mask, device=module.E_train_mask.device, requires_grad=False)
#            m[:k] = 1.
#        
#        E = module.E.to(module.E_train_mask.dtype).to(module.E_train_mask.device)
#        loss += torch.sum((module.E_train_mask * E - m * E)**2)
#
#    loss = loss/len(compression_calculator.lowrank_layers)
#   return loss

class LowrankLinearSimple(torch.nn.Module):
    def __init__(self, current_layer, svd_vector, alpha=1., niter=2, tau=0.4):
        """
        Linear estimator for the mask
        """
        super(LowrankLinearSimple, self).__init__()

        if not isinstance(current_layer, torch.nn.Linear):
            raise ValueError(f"Expected input into SVDLayer be of instance nn.Linear, got {type(current_layer)}")
        
        # bias to add to gumbel sigmoid to ensure full rank is selected
        dtype = current_layer.weight.dtype
        self.in_features, self.out_features = current_layer.in_features, current_layer.out_features
        self.rank = min(current_layer.weight.shape[1], current_layer.weight.shape[0])

        weight = current_layer.weight.float()
        layer_device = weight.device

        if torch.cuda.is_available():
            weight = weight.cuda()

        if svd_vector is not None: 
            svd_vector += 1e-6 # division by zero
            svd_vector = svd_vector.to(weight.device)**alpha
            weight = weight * svd_vector.unsqueeze(0)
        
        with torch.no_grad():
            U, E, V = torch.svd_lowrank(weight, 
                                        q=self.rank,
                                        niter=niter)
        
        if svd_vector is not None: 
            V = V / svd_vector.unsqueeze(1)
        
        U, E, V = U.to(layer_device), E.to(layer_device), V.to(layer_device)

        assert len(E.shape) == 1, 'expected singular values to have only one dim'

        # precompute EV for efficency
        self.UE = torch.nn.Parameter((U * E.unsqueeze(0)).to(dtype), requires_grad=True)
        self.V_t = torch.nn.Parameter(V.T.to(dtype), requires_grad=True)
        self.E = E
        self.E.requires_grad=False

        init_vector = torch.linspace(6, 3., len(E), device=E.device).to(dtype)
        self.E_train = torch.nn.Parameter(init_vector)

        self.tau = tau
        self.use_stored_masks = False
        self.E_train_mask = None 

    def forward(self, inputs):
        """
        Computes forward pass with selection of singular values through predicted mask.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        E_train_mask = self.calculate_mask(is_training=self.training)
        if self.training:
            self.E_train_mask = E_train_mask

        if inputs.device != self.V_t.device: # multi-gpu setup
            inputs = inputs.to(self.V_t.device)
        
        inputs = inputs.transpose(1, 2)
        output = (self.UE * E_train_mask.unsqueeze(0)) @ (self.V_t @ inputs)
        output = output.transpose(1, 2)

        return output
    
    def calculate_mask(self, is_training, return_topk=False):
        """
        Calculates the mask for singular value selection. During training, it uses Gumbel-Sigmoid approximation.
        During evaluation, various non-differentiable operations can be used

        Args:
            is_training (bool): Whether the model is in training mode.
            return_probs (bool): Whether to return probabilities instead of binary mask (default: False).

        Returns:
            torch.Tensor: Mask for singular value selection.
        """

        if is_training or self.E_train_mask is None:
            logit_mask = self.E_train
            E_train_mask = gumbel_sigmoid(logit_mask, tau=self.tau)
            E_train_mask = STE.apply(E_train_mask)
        else:
            E_train_mask = self.E_train_mask

        if not is_training:
            E_train_mask = E_train_mask > 0.5
            topk_mask = torch.zeros_like(E_train_mask, device=E_train_mask.device, requires_grad=False)
            topk_mask[:E_train_mask.sum().item()] = 1.
            return topk_mask

        return E_train_mask

    def __str__(self):
        return f"LowrankLinearSimple(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank})"

    def __repr__(self):
        return self.__str__()
    
def calculate_r_align(compression_calculator):
    """
    Loss to align learned mask to singular value properties
    """
    loss = 0.
    for module in compression_calculator.lowrank_layers: 
        with torch.no_grad():
            m = module.calculate_mask(False).detach()
            #k = module.E_train_mask.detach().sum().item()
            #m = torch.zeros_like(module.E_train_mask, device=module.E_train_mask.device, requires_grad=False)
            #m[:k] = 1.
        
        E = module.E.to(module.E_train_mask.dtype).to(module.E_train_mask.device)

        if isinstance(loss, torch.Tensor): 
            loss += F.mse_loss(module.E_train_mask * E, m * E, reduction='mean').to(loss.device)
        else:
            loss += F.mse_loss(module.E_train_mask * E, m * E, reduction='mean')

    loss = loss/len(compression_calculator.lowrank_layers)
    return loss

def calculate_R_loss(compression_calculator, target_param_ratio:int):
    """
    Compression regularizer
    
    """
    total_new_params = 0. 
    total_orignal_params = 0.
    for module in compression_calculator.lowrank_layers: 
        mask_sum = module.E_train_mask.sum()

        if isinstance(total_new_params, torch.Tensor): 
            mask_sum = mask_sum.to(total_new_params.device) # for multigpu
    
        total_new_params += (module.in_features + module.out_features) * mask_sum
        total_orignal_params += (module.in_features * module.out_features)

    target_params = target_param_ratio * total_orignal_params

    a = total_new_params/total_orignal_params
    if total_new_params.item() < target_params:
        a = torch.tensor(target_param_ratio)
    
    loss = torch.log(a/target_param_ratio)
    return loss

def calculate_R_loss_simple(compression_calculator):
    """
    Simple compression regularizer, that minimizes the mean of the mask (not from ASR paper)
    
    """
    loss = 0. 
    for module in compression_calculator.lowrank_layers: 
        mask_mean = module.E_train.mean()

        if isinstance(loss, torch.Tensor): 
            mask_mean = mask_mean.to(loss.device) # for multigpu
    
        loss += mask_mean

    return loss/len(compression_calculator.lowrank_layers)

def training_step(model, batch, pad_token_id, args, compression_calculator, is_eval=False):
    """
    One training step of model
    """

    # create inputs and targets
    input_ids = batch['input_ids'][:, :-1].to(model.device)
    attention_mask = batch['attention_mask'][:, :-1].to(model.device)
    labels = batch['input_ids'][:, 1:].clone().to(model.device)
    labels[labels == pad_token_id] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits

    logits_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='mean', ignore_index=-100)

    with torch.no_grad():
        perplexity = torch.exp(logits_loss)

    if is_eval: 
        return None, logits_loss, None, None, perplexity, None, None, None

    
    r_align_loss = calculate_r_align(compression_calculator)
    
    if args.r_loss == 'default': 
        r_loss = calculate_R_loss(compression_calculator, args.p_param)
    elif args.r_loss == 'simple':
        r_loss = calculate_R_loss_simple(compression_calculator)

    with torch.no_grad():
        current_param_ratio, keep_ratio = compression_calculator.get_compression_and_sv()
        #current_param_ratio = compression_calculator.get_compression()
        #keep_ratio = compression_calculator.get_sv_ratio()

    # if compression is reached, ignore compression regularizer
    lambda_scale = args.lambda_scale
    if abs(current_param_ratio - args.target_param_ratio) < 0.002: 
        lambda_scale = 0

    loss = args.beta_scale * logits_loss + lambda_scale * r_loss + args.gamma_scale * r_align_loss

    return loss, logits_loss, r_align_loss, r_loss, perplexity, keep_ratio, current_param_ratio, lambda_scale

def eval_model(model, test_dl, pad_token_id, args, compression_calculator):
    """
    Perform evaluation
    """
    model = model.eval()
    metrics = defaultdict(list)
    for _, batch in enumerate(tqdm(test_dl, desc=f"Evaluating", mininterval=5)):
        with torch.no_grad():
            loss, logits_loss, r_align_loss, r_loss, perplexity, keep_ratio, current_param_ratio, lambda_scale = training_step(model, batch, pad_token_id, args, compression_calculator, is_eval=True)

        metrics['logits_loss'].append(logits_loss.item() if isinstance(logits_loss, torch.Tensor) else logits_loss)
        metrics['perplexity'].append(perplexity.item())

    for key in metrics:
        metrics[key] = sum(metrics[key]) / len(metrics[key])

    metrics = {f"eval/{key}": value for key, value in metrics.items()}
    torch.cuda.empty_cache()
    model = model.train()
    return metrics


def freeze_model_masks(model, should_freeze=True):
    """ 
    Freezes masks so that the model does not generate a mask in the forward pass, but uses a pre-computed mask 
    """
    for _, module in model.named_modules():
        if 'Lowrank' in str(module)[:7]:
            module.use_stored_masks = should_freeze

class HypernetOperator:
    """
    Handles other operations required for hypenetwork
    """
    def __init__(self, lowrank_layers):
        self.L = len(lowrank_layers)
        self.hypernet = Hypernet_GRU(self.L, 32, 64)
        self.lowrank_layers = lowrank_layers

    def update_network_with_sv_hidden_state(self):
        hidden = self.hypernet()
        for i in range(self.L):
            self.lowrank_layers[i].global_hypernet_state = hidden[i, :]
