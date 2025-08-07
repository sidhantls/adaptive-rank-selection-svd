"""
Fast Robust Principal Component Analysis (RPCA) implementation.
"""
import numpy as np
from typing import Tuple, Optional


def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """
    Soft thresholding operator.
    
    Args:
        X: Input matrix
        tau: Threshold parameter
        
    Returns:
        Soft thresholded matrix
    """
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def singular_value_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """
    Singular value thresholding operator.
    
    Args:
        X: Input matrix
        tau: Threshold parameter
        
    Returns:
        Matrix with thresholded singular values
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0)
    return U @ np.diag(s_thresh) @ Vt


def fast_rpca_alm(M: np.ndarray, lambda_param: Optional[float] = None,
                  mu: float = 1e-3, max_iterations: int = 1000,
                  tolerance: float = 1e-7,
                  verbose: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], int]:
    """
    Fast Robust PCA using Augmented Lagrange Multipliers (ALM).
    
    Solves:
    minimize ||L||_* + λ||S||_1
    subject to L + S = M
    
    Args:
        M: Input matrix
        lambda_param: Sparsity parameter (if None, set to 1/sqrt(max(m,n)))
        mu: Penalty parameter for augmented Lagrangian
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        verbose: Whether to print progress
        
    Returns:
        Tuple containing:
        - (L, S): Low rank and sparse decomposition
        - Number of iterations
    """
    m, n = M.shape
    
    if lambda_param is None:
        lambda_param = 1.0 / np.sqrt(max(m, n))
    
    # Initialize variables
    L = np.zeros((m, n))
    S = np.zeros((m, n))
    Y = np.zeros((m, n))  # Lagrange multipliers
    
    mu_inv = 1.0 / mu
    
    # Precompute Frobenius norm of M
    norm_M = np.linalg.norm(M, 'fro')
    
    for iteration in range(max_iterations):
        # Update L using singular value thresholding
        L = singular_value_threshold(M - S + mu_inv * Y, mu_inv)
        
        # Update S using soft thresholding
        S = soft_threshold(M - L + mu_inv * Y, lambda_param * mu_inv)
        
        # Update Lagrange multipliers
        Y = Y + mu * (M - L - S)
        
        # Check convergence
        primal_residual = M - L - S
        primal_residual_norm = np.linalg.norm(primal_residual, 'fro')
        
        if primal_residual_norm / norm_M < tolerance:
            if verbose:
                print(f"Fast RPCA converged in {iteration + 1} iterations")
            break
        
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: Residual = {primal_residual_norm / norm_M:.6f}")
    
    return (L, S), iteration + 1


def fast_rpca_ialm(M: np.ndarray, lambda_param: Optional[float] = None,
                   max_iterations: int = 1000, tolerance: float = 1e-7,
                   verbose: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], int]:
    """
    Fast Robust PCA using Inexact Augmented Lagrange Multipliers (IALM).
    
    Args:
        M: Input matrix
        lambda_param: Sparsity parameter
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        verbose: Print progress
        
    Returns:
        Tuple containing:
        - (L, S): Low rank and sparse decomposition
        - Number of iterations
    """
    m, n = M.shape
    
    if lambda_param is None:
        lambda_param = 1.0 / np.sqrt(max(m, n))
    
    # Initialize
    L = np.zeros((m, n))
    S = np.zeros((m, n))
    Y = np.zeros((m, n))
    
    # Parameters
    mu = 1.25 / np.linalg.norm(M, 2)  # Spectral norm
    mu_bar = mu * 1e7
    rho = 1.5
    
    norm_M = np.linalg.norm(M, 'fro')
    
    for iteration in range(max_iterations):
        # Update L
        temp = M - S + Y / mu
        L = singular_value_threshold(temp, 1.0 / mu)
        
        # Update S
        temp = M - L + Y / mu
        S = soft_threshold(temp, lambda_param / mu)
        
        # Check convergence before updating multipliers
        primal_residual = M - L - S
        primal_residual_norm = np.linalg.norm(primal_residual, 'fro')
        
        if primal_residual_norm / norm_M < tolerance:
            if verbose:
                print(f"Fast RPCA-IALM converged in {iteration + 1} iterations")
            break
        
        # Update multipliers
        Y = Y + mu * primal_residual
        
        # Update penalty parameter
        if primal_residual_norm > 0.9 * np.linalg.norm(Y / mu, 'fro'):
            mu = min(rho * mu, mu_bar)
        
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: Residual = {primal_residual_norm / norm_M:.6f}, mu = {mu:.6f}")
    
    return (L, S), iteration + 1


def fast_rpca_with_constraints(M: np.ndarray, k_rank: int, k_sparse: int,
                              lambda_param: Optional[float] = None,
                              method: str = 'ialm',
                              max_iterations: int = 1000,
                              tolerance: float = 1e-7) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """
    Fast RPCA with explicit rank and sparsity constraints.
    
    Args:
        M: Input matrix
        k_rank: Maximum rank constraint
        k_sparse: Maximum sparsity constraint
        lambda_param: Sparsity regularization parameter
        method: 'alm' or 'ialm'
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple containing:
        - (L, S): Constrained low rank and sparse matrices
        - Reconstruction error
    """
    # First solve unconstrained RPCA
    if method == 'ialm':
        (L_rpca, S_rpca), _ = fast_rpca_ialm(M, lambda_param, max_iterations, tolerance)
    else:
        (L_rpca, S_rpca), _ = fast_rpca_alm(M, lambda_param, max_iterations=max_iterations, tolerance=tolerance)
    
    # Project to rank constraint
    U, s, Vt = np.linalg.svd(L_rpca, full_matrices=False)
    s_truncated = s.copy()
    s_truncated[k_rank:] = 0
    L_constrained = U @ np.diag(s_truncated) @ Vt
    
    # Project to sparsity constraint
    S_flat = S_rpca.flatten()
    indices = np.argsort(np.abs(S_flat))[::-1]
    S_constrained_flat = np.zeros_like(S_flat)
    S_constrained_flat[indices[:k_sparse]] = S_flat[indices[:k_sparse]]
    S_constrained = S_constrained_flat.reshape(S_rpca.shape)
    
    # Compute reconstruction error
    reconstruction_error = np.linalg.norm(M - L_constrained - S_constrained, 'fro')**2
    
    return (L_constrained, S_constrained), reconstruction_error


def robust_pca_admm(M: np.ndarray, lambda_param: float, 
                   rho: float = 1.0, max_iterations: int = 1000,
                   tolerance: float = 1e-6) -> Tuple[Tuple[np.ndarray, np.ndarray], int]:
    """
    Robust PCA using Alternating Direction Method of Multipliers (ADMM).
    
    Args:
        M: Input matrix
        lambda_param: Sparsity parameter
        rho: ADMM penalty parameter
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple containing:
        - (L, S): Low rank and sparse decomposition
        - Number of iterations
    """
    m, n = M.shape
    
    # Initialize variables
    L = np.zeros((m, n))
    S = np.zeros((m, n))
    Z = np.zeros((m, n))
    U = np.zeros((m, n))
    
    for iteration in range(max_iterations):
        # L-minimization step
        L = singular_value_threshold(M - S - Z + U / rho, 1.0 / rho)
        
        # S-minimization step
        S = soft_threshold(M - L - Z + U / rho, lambda_param / rho)
        
        # Z-minimization step (just projection onto constraint set)
        Z = M - L - S
        
        # Dual variable update
        U = U + rho * (M - L - S - Z)
        
        # Check convergence
        primal_residual = np.linalg.norm(M - L - S - Z, 'fro')
        dual_residual = rho * np.linalg.norm(Z, 'fro')
        
        if max(primal_residual, dual_residual) < tolerance:
            break
    
    return (L, S), iteration + 1


def accelerated_rpca(M: np.ndarray, lambda_param: Optional[float] = None,
                    max_iterations: int = 1000, tolerance: float = 1e-7) -> Tuple[Tuple[np.ndarray, np.ndarray], int]:
    """
    Accelerated RPCA using Nesterov-like acceleration.
    
    Args:
        M: Input matrix
        lambda_param: Sparsity parameter
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple containing:
        - (L, S): Low rank and sparse decomposition
        - Number of iterations
    """
    m, n = M.shape
    
    if lambda_param is None:
        lambda_param = 1.0 / np.sqrt(max(m, n))
    
    # Initialize
    L = np.zeros((m, n))
    S = np.zeros((m, n))
    L_prev = L.copy()
    S_prev = S.copy()
    
    t = 1.0
    mu = 0.25 / np.linalg.norm(M, 2)
    
    norm_M = np.linalg.norm(M, 'fro')
    
    for iteration in range(max_iterations):
        # Acceleration
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        beta = (t - 1) / t_new
        
        # Momentum updates
        L_momentum = L + beta * (L - L_prev)
        S_momentum = S + beta * (S - S_prev)
        
        # Store previous iterates
        L_prev, S_prev = L.copy(), S.copy()
        
        # Proximal operators with momentum
        temp_L = M - S_momentum
        L = singular_value_threshold(temp_L, 1.0 / mu)
        
        temp_S = M - L
        S = soft_threshold(temp_S, lambda_param / mu)
        
        # Update t
        t = t_new
        
        # Check convergence
        residual = np.linalg.norm(M - L - S, 'fro')
        if residual / norm_M < tolerance:
            break
    
    return (L, S), iteration + 1
