"""
Stable Principal Component Pursuit (SPCP) implementation.
"""
import numpy as np
import cvxpy as cp
from typing import Tuple, Optional


def stable_principal_component_pursuit(M: np.ndarray, sigma: float,
                                     threshold: float = 1e-6,
                                     solver_output: bool = False,
                                     solver: str = 'SCS') -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """
    Solve the stable principal component pursuit problem.
    
    This solves:
    minimize    ||L||_* + λ||S||_1 + (1/2μ)||M - L - S||_F^2
    
    where λ = σ/√(2n) and μ = σ√(2n)
    
    Args:
        M: Input matrix
        sigma: Parameter controlling tradeoff between nuclear norm and L1 norm
        threshold: Values below threshold are set to zero
        solver_output: Whether to display solver output
        solver: CVXPY solver to use
        
    Returns:
        Tuple containing:
        - (L, S): Low rank and sparse matrix decomposition
        - Optimal objective value
    """
    n = M.shape[0]
    lambda_param = sigma / np.sqrt(2 * n)
    mu = sigma * np.sqrt(2 * n)
    
    # Define variables
    L = cp.Variable(M.shape)
    S = cp.Variable(M.shape)
    
    # Define objective
    nuclear_norm = cp.norm(L, "nuc")
    l1_norm = cp.norm(S, 1)
    frobenius_term = cp.sum_squares(M - L - S) / (2 * mu)
    
    objective = cp.Minimize(nuclear_norm + lambda_param * l1_norm + frobenius_term)
    
    # Create and solve problem
    prob = cp.Problem(objective)
    
    try:
        if solver == 'MOSEK':
            prob.solve(solver=cp.MOSEK, verbose=solver_output)
        elif solver == 'SCS':
            prob.solve(solver=cp.SCS, verbose=solver_output)
        else:
            prob.solve(verbose=solver_output)
    except Exception as e:
        print(f"SPCP solver failed: {e}")
        return (np.zeros_like(M), np.zeros_like(M)), float('inf')
    
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print(f"SPCP solver status: {prob.status}")
        return (np.zeros_like(M), np.zeros_like(M)), float('inf')
    
    # Extract solution and apply thresholding
    L_opt = L.value if L.value is not None else np.zeros_like(M)
    S_opt = S.value if S.value is not None else np.zeros_like(M)
    
    # Apply thresholding
    L_opt[np.abs(L_opt) < threshold] = 0
    S_opt[np.abs(S_opt) < threshold] = 0
    
    return (L_opt, S_opt), prob.value


def spcp_with_rank_sparsity_constraints(M: np.ndarray, k_rank: int, k_sparse: int,
                                       sigma: float = 1.0,
                                       threshold: float = 1e-6) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """
    SPCP with post-processing to satisfy rank and sparsity constraints.
    
    Args:
        M: Input matrix
        k_rank: Maximum rank constraint
        k_sparse: Maximum sparsity constraint
        sigma: SPCP parameter
        threshold: Thresholding parameter
        
    Returns:
        Tuple containing:
        - (L, S): Constrained low rank and sparse matrices
        - Reconstruction error
    """
    # Solve SPCP
    (L_spcp, S_spcp), _ = stable_principal_component_pursuit(M, sigma, threshold)
    
    # Project L to rank constraint
    U, s, Vt = np.linalg.svd(L_spcp, full_matrices=False)
    s_truncated = s.copy()
    s_truncated[k_rank:] = 0
    L_projected = U @ np.diag(s_truncated) @ Vt
    
    # Project S to sparsity constraint
    S_flat = S_spcp.flatten()
    indices = np.argsort(np.abs(S_flat))[::-1]
    S_projected = np.zeros_like(S_spcp)
    S_flat_proj = np.zeros_like(S_flat)
    S_flat_proj[indices[:k_sparse]] = S_flat[indices[:k_sparse]]
    S_projected = S_flat_proj.reshape(S_spcp.shape)
    
    # Compute reconstruction error
    reconstruction_error = np.linalg.norm(M - L_projected - S_projected, 'fro')**2
    
    return (L_projected, S_projected), reconstruction_error
