import torch
from utils.lowrank_methods.alternating_minimization import SLR_AM
from utils.adaptive_rank_selection import gumbel_sigmoid, STE

class LowrankLinearSimpleSLR(torch.nn.Module):
    def __init__(self, current_layer, svd_vector, alpha=1., niter=2, tau=0.4,
                 k_sparse_ratio=0.1, mu=1.0, lambda_reg=1.0, corr_rank=None):
        """
        Linear estimator for the mask using SLR_AM (Sparse Low-Rank Alternating Minimization)
        with PyTorch tensor support.
        
        Args:
            current_layer: The original linear layer to decompose
            svd_vector: Optional weighting vector for activation/loss aware decomposition
            alpha: Power to raise the svd_vector to
            niter: Number of iterations for decomposition methods
            tau: Temperature parameter for gumbel sigmoid
            k_sparse_ratio: Ratio of entries to keep in the sparse component (as a ratio of total entries)
            mu: Regularization parameter for sparse matrix in SLR_AM
            lambda_reg: Regularization parameter for low rank matrix in SLR_AM
        """
        super(LowrankLinearSimpleSLR, self).__init__()

        if not isinstance(current_layer, torch.nn.Linear):
            raise ValueError(f"Expected input into SLR_AM Layer to be of instance nn.Linear, got {type(current_layer)}")
        
        dtype = current_layer.weight.dtype
        self.in_features, self.out_features = current_layer.in_features, current_layer.out_features
        self.rank = min(current_layer.weight.shape[1], current_layer.weight.shape[0])

        weight = current_layer.weight.float()
        layer_device = weight.device

        if torch.cuda.is_available():
            weight = weight.cuda()

        # Apply activation/loss aware scaling if provided
        if svd_vector is not None: 
            svd_vector += 1e-6  # Avoid division by zero
            svd_vector = svd_vector.to(weight.device)**alpha
            weight = weight * svd_vector.unsqueeze(0)
        
        # Calculate number of sparse elements to keep
        total_elements = weight.shape[0] * weight.shape[1]
        k_sparse = int(k_sparse_ratio * total_elements)
        
        # Run SLR_AM to get low-rank and sparse components (now expects PyTorch tensors)
        with torch.no_grad():
            (X_lr, Y_sparse), _ = SLR_AM(
                weight,
                mu=mu,
                lambda_reg=lambda_reg,
                k_sparse=k_sparse,
                k_rank=self.rank,
                device=str(weight.device),
                random_restarts=1,
                exact_svd=True
            )
        # Ensure tensors are on the correct device
        X_lr = X_lr.to(layer_device)
        Y_sparse = Y_sparse.to(layer_device)
        # Perform SVD on the low-rank component to get U, E, V
        U_lr, E_lr, V_lr = torch.svd_lowrank(X_lr, q=self.rank, niter=niter)
        # Factorize sparse correction into low-rank of size corr_rank
        corr_rank = corr_rank if corr_rank is not None else min(self.rank, 10)
        U_s, E_s, V_s = torch.svd_lowrank(Y_sparse, q=corr_rank, niter=niter)
        # Apply inverse scaling if needed
        if svd_vector is not None:
            svd_vector_device = svd_vector.to(V_lr.device)
            V_lr = V_lr / svd_vector_device.unsqueeze(1)
            V_s = V_s / svd_vector_device.unsqueeze(1)
        # Move to correct device and dtype
        U_lr, E_lr, V_lr = U_lr.to(layer_device), E_lr.to(layer_device), V_lr.to(layer_device)
        U_s, E_s, V_s = U_s.to(layer_device), E_s.to(layer_device), V_s.to(layer_device)
        # Combine original low-rank and sparse-correction factors
        # UE block: [U_lr*E_lr, U_s*E_s]
        UE_lr = (U_lr * E_lr.unsqueeze(0)).to(dtype)
        UE_s = (U_s * E_s.unsqueeze(0)).to(dtype)
        self.UE = torch.nn.Parameter(torch.cat([UE_lr, UE_s], dim=1), requires_grad=True)
        # V_t block: vertical stack of V_lr^T and V_s^T
        Vt_lr = V_lr.T.to(dtype)
        Vt_s = V_s.T.to(dtype)
        self.V_t = torch.nn.Parameter(torch.cat([Vt_lr, Vt_s], dim=0), requires_grad=True)
        # Store mask dims
        self.orig_rank = self.rank
        self.corr_rank = corr_rank
        # Initialize learnable parameters for singular value selection on original dims
        init_vector = torch.linspace(6, 3., self.orig_rank, device=E_lr.device).to(dtype)
        self.E_train = torch.nn.Parameter(init_vector)
        # Total combined rank
        self.total_rank = self.orig_rank + self.corr_rank
        # Keep E for debug
        self.E = E_lr
        self.E.requires_grad = False
        
        self.tau = tau
        self.use_stored_masks = False
        self.E_train_mask = None

    def forward(self, inputs):
        """
        Computes forward pass with selection of singular values through predicted mask.
        Also adds the sparse component to the output.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Compute mask for original singular dims and include all correction dims
        base_mask = self.calculate_mask(is_training=self.training)
        ones_corr = torch.ones(self.corr_rank, device=base_mask.device)
        E_train_mask = torch.cat([base_mask, ones_corr], dim=0)
        if self.training:
            self.E_train_mask = E_train_mask

        if inputs.device != self.V_t.device:  # multi-gpu setup
            inputs = inputs.to(self.V_t.device)
        
        # Handle 2D and 3D inputs appropriately
        # Single two-stage matmul: (UE_masked) @ (V_t @ x)
        if inputs.dim() == 3:
            x = inputs.transpose(1, 2)
            y = (self.UE * E_train_mask.unsqueeze(0)) @ (self.V_t @ x)
            output = y.transpose(1, 2)
        else:
            x = inputs.T
            y = (self.UE * E_train_mask.unsqueeze(0)) @ (self.V_t @ x)
            output = y.T
        return output
    
    def calculate_mask(self, is_training, return_topk=False):
        """
        Calculates the mask for singular value selection. During training, it uses Gumbel-Sigmoid approximation.
        During evaluation, various non-differentiable operations can be used.

        Args:
            is_training (bool): Whether the model is in training mode.
            return_topk (bool): Whether to return top-k mask.

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
        return f"LowrankLinearSimpleSLR(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank})"
