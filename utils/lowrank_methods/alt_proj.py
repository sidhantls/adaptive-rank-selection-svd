"""
Alternating Projection method for sparse plus low rank matrix decomposition.
"""
import numpy as np
from typing import Tuple


def hard_threshold(A: np.ndarray, threshold_param: float) -> np.ndarray:
    """
    Perform hard thresholding of matrix A.
    
    Args:
        A: Input matrix
        threshold_param: Hard thresholding value
        
    Returns:
        Hard thresholded matrix
    """
    filter_mat = np.abs(A) >= threshold_param
    return A * filter_mat


def alternating_projection(A: np.ndarray, k_rank: int,
                         beta_mult: float = 1.0, zeta_mult: float = 1.0,
                         epsilon: float = 1e-3,
                         max_iterations: int = 1000) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """
    Alternating projection algorithm for sparse plus low rank matrix decomposition.
    
    Args:
        A: Input matrix
        k_rank: Maximum rank of low rank component
        beta_mult: Multiplier for beta parameter
        zeta_mult: Multiplier for zeta parameter  
        epsilon: Termination criterion
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple containing:
        - (X, Y): Low rank and sparse matrix decomposition
        - Final objective value
    """
    n = A.shape[0]
    
    # Compute SVD of input matrix
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Compute "empirical" incoherence parameter
    incoherence_param = 0
    for i in range(n):
        incoherence_param = max([
            np.linalg.norm(U[i, :]) * np.sqrt(n / k_rank),
            np.linalg.norm(Vt[:, i]) * np.sqrt(n / k_rank),
            incoherence_param
        ])
    
    # Initialize algorithm parameters
    beta = beta_mult * 4 * incoherence_param**2 * k_rank / n
    current_threshold = beta * sigma[0]
    
    # Initialize iterates
    X_iterate = np.zeros((n, n))
    Y_iterate = np.zeros((n, n))
    
    # Set initial sparse component using hard thresholding
    Y_iterate = hard_threshold(A, current_threshold)
    
    prev_objective = np.inf
    
    for iteration in range(max_iterations):
        # Project (A - Y) onto low rank matrices
        residual = A - Y_iterate
        U_res, s_res, Vt_res = np.linalg.svd(residual, full_matrices=False)
        
        # Keep only top k_rank components
        s_truncated = s_res.copy()
        s_truncated[k_rank:] = 0
        X_iterate = U_res @ np.diag(s_truncated) @ Vt_res
        
        # Update sparse component using hard thresholding
        residual = A - X_iterate
        Y_iterate = hard_threshold(residual, current_threshold)
        
        # Compute objective (reconstruction error)
        objective = np.linalg.norm(A - X_iterate - Y_iterate, 'fro')**2
        
        # Check convergence
        if abs(prev_objective - objective) / max(prev_objective, 1e-12) < epsilon:
            break
            
        prev_objective = objective
        
        # Adaptive threshold update (optional)
        if iteration > 10:
            current_threshold *= 0.99
    
    return (X_iterate, Y_iterate), objective


def alternating_projection_with_sparsity(A: np.ndarray, k_rank: int, k_sparse: int,
                                       beta_mult: float = 1.0,
                                       epsilon: float = 1e-3,
                                       max_iterations: int = 1000) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """
    Alternating projection with explicit sparsity constraint.
    
    Args:
        A: Input matrix
        k_rank: Maximum rank of low rank component
        k_sparse: Maximum sparsity of sparse component
        beta_mult: Multiplier for beta parameter
        epsilon: Termination criterion
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple containing:
        - (X, Y): Low rank and sparse matrix decomposition
        - Final objective value
    """
    n = A.shape[0]
    
    # Initialize iterates
    X_iterate = np.zeros((n, n))
    Y_iterate = np.zeros((n, n))
    
    prev_objective = np.inf
    
    for iteration in range(max_iterations):
        # Project (A - Y) onto rank-k matrices
        residual = A - Y_iterate
        U_res, s_res, Vt_res = np.linalg.svd(residual, full_matrices=False)
        
        # Keep only top k_rank components
        s_truncated = s_res.copy()
        s_truncated[k_rank:] = 0
        X_iterate = U_res @ np.diag(s_truncated) @ Vt_res
        
        # Project (A - X) onto k-sparse matrices
        residual = A - X_iterate
        
        # Find k largest entries in absolute value
        residual_flat = residual.flatten()
        indices = np.argsort(np.abs(residual_flat))[::-1]
        
        Y_flat = np.zeros_like(residual_flat)
        Y_flat[indices[:k_sparse]] = residual_flat[indices[:k_sparse]]
        Y_iterate = Y_flat.reshape(residual.shape)
        
        # Compute objective
        objective = np.linalg.norm(A - X_iterate - Y_iterate, 'fro')**2
        
        # Check convergence
        if abs(prev_objective - objective) / max(prev_objective, 1e-12) < epsilon:
            break
            
        prev_objective = objective
    
    return (X_iterate, Y_iterate), objective


def robust_alternating_projection(A: np.ndarray, k_rank: int, k_sparse: int,
                                lambda_reg: float = 0.1, mu_reg: float = 0.1,
                                epsilon: float = 1e-3,
                                max_iterations: int = 1000) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """
    Robust alternating projection with regularization.
    
    Args:
        A: Input matrix
        k_rank: Maximum rank constraint
        k_sparse: Maximum sparsity constraint
        lambda_reg: Regularization for low rank component
        mu_reg: Regularization for sparse component
        epsilon: Convergence criterion
        max_iterations: Maximum iterations
        
    Returns:
        Tuple containing:
        - (X, Y): Low rank and sparse matrices
        - Final objective value
    """
    n = A.shape[0]
    
    # Initialize with random feasible solution
    X_iterate = np.random.randn(n, n)
    U, s, Vt = np.linalg.svd(X_iterate, full_matrices=False)
    s[k_rank:] = 0
    X_iterate = U @ np.diag(s) @ Vt
    
    Y_iterate = np.random.randn(n, n)
    Y_flat = Y_iterate.flatten()
    indices = np.argsort(np.abs(Y_flat))[::-1]
    Y_flat[indices[k_sparse:]] = 0
    Y_iterate = Y_flat.reshape((n, n))
    
    prev_objective = np.inf
    
    for iteration in range(max_iterations):
        # Update X (low rank component)
        residual = A - Y_iterate
        U_res, s_res, Vt_res = np.linalg.svd(residual, full_matrices=False)
        
        # Soft thresholding for regularization
        s_thresh = np.maximum(s_res - lambda_reg, 0)
        s_thresh[k_rank:] = 0  # Hard rank constraint
        
        X_iterate = U_res @ np.diag(s_thresh) @ Vt_res
        
        # Update Y (sparse component)
        residual = A - X_iterate
        
        # Soft thresholding followed by hard sparsity constraint
        Y_soft = np.sign(residual) * np.maximum(np.abs(residual) - mu_reg, 0)
        
        # Project to k-sparse
        Y_flat = Y_soft.flatten()
        indices = np.argsort(np.abs(Y_flat))[::-1]
        Y_proj_flat = np.zeros_like(Y_flat)
        Y_proj_flat[indices[:k_sparse]] = Y_flat[indices[:k_sparse]]
        Y_iterate = Y_proj_flat.reshape(residual.shape)
        
        # Compute objective with regularization
        reconstruction_error = np.linalg.norm(A - X_iterate - Y_iterate, 'fro')**2
        reg_term = lambda_reg * np.linalg.norm(X_iterate, 'fro')**2 + mu_reg * np.linalg.norm(Y_iterate, 'fro')**2
        objective = reconstruction_error + reg_term
        
        # Check convergence
        if abs(prev_objective - objective) / max(prev_objective, 1e-12) < epsilon:
            break
            
        prev_objective = objective
    
    return (X_iterate, Y_iterate), reconstruction_error
