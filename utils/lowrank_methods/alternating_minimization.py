"""
SLR-AM in PyTorch
Alternating minimization for sparse-plus-low-rank matrix decomposition
with optional GPU acceleration.

Author: <your-name>, 2025-08-06
"""

from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor

# ---------- helpers ---------- #

def _to(x: Tensor, device):
    "Move tensor to device if it is not already there."
    return x if x.device == device else x.to(device)

def compute_objective_value(
        X: Tensor, Y: Tensor, U: Tensor,
        mu: float, lambda_reg: float
) -> float:
    """ ||U-X-Y||_F² + λ||X||_F² + μ||Y||_F² (returned as python float) """
    loss = (U - X - Y).norm()**2
    loss += lambda_reg * X.norm()**2 + mu * Y.norm()**2
    return loss.item()

def is_feasible(
        X: Tensor, Y: Tensor,
        k_sparse: int, k_rank: int,
        eps: float = 1e-10
) -> bool:
    nonzeros = (Y.abs() > eps).sum().item()
    rank = torch.linalg.matrix_rank(X).item()
    return nonzeros <= k_sparse and rank <= k_rank

# ---------- low-rank projection ---------- #

def project_matrix(
        A: Tensor, k_rank: int,
        exact_svd: bool = True
) -> Tensor:
    """
    Truncate A to rank-k using SVD; works on GPU.
    For tall or wide matrices, torch.svd_lowrank (randomized) is faster.
    """
    m, n = A.shape
    if k_rank >= min(m, n):
        return A

    if exact_svd:
        # full or truncated based on shape
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        return (U[:, :k_rank] * S[:k_rank]) @ Vh[:k_rank]
    else:
        # randomized SVD with power iterations=2
        U, S, Vh = torch.linalg.svd_lowrank(A, q=k_rank, niter=2)
        return (U * S) @ Vh

# ---------- sparsity pattern ---------- #

def construct_binary_matrix(
        A: Tensor, k_sparse: int,
        zero_idx: Optional[List[Tuple[int,int]]] = None,
        one_idx: Optional[List[Tuple[int,int]]] = None
) -> Tensor:
    """
    Return 0/1 mask S identifying k_sparse largest |A_ij| values,
    honoring forced zeros/ones.
    """
    zero_idx = zero_idx or []
    one_idx  = one_idx  or []

    S = torch.zeros_like(A, dtype=torch.bool)
    if one_idx:
        rows, cols = zip(*one_idx)
        S[list(rows), list(cols)] = True

    remaining = k_sparse - len(one_idx)
    if remaining <= 0:
        return S

    # mask out forbidden entries
    mask = torch.ones_like(A, dtype=torch.bool)
    if zero_idx:
        r0, c0 = zip(*zero_idx)
        mask[list(r0), list(c0)] = False
    if one_idx:
        mask[list(rows), list(cols)] = False

    # take top-k remaining
    vals = A.abs()[mask]
    if vals.numel() <= remaining:
        S[mask] = True
        return S
    thresh = torch.kthvalue(vals, vals.numel() - remaining + 1).values
    S &= False  # reset dynamic region
    S[mask & (A.abs() >= thresh)] = True
    return S

# ---------- sub-problems ---------- #

def solve_sparse_problem(
        U_tilde: Tensor, mu: float, k_sparse: int,
        zero_idx=None, one_idx=None
) -> Tensor:
    S = construct_binary_matrix(U_tilde, k_sparse, zero_idx, one_idx)
    return (S * U_tilde) / (1.0 + mu)

def solve_rank_problem(
        U_tilde: Tensor, lambda_reg: float, k_rank: int,
        exact_svd: bool = True
) -> Tensor:
    return project_matrix(U_tilde / (1.0 + lambda_reg), k_rank, exact_svd)

# ---------- alternating loop ---------- #

def iterate_X_Y(
        U: Tensor, mu: float, lambda_reg: float,
        k_sparse: int, k_rank: int,
        X_init: Tensor, Y_init: Tensor,
        zero_idx=None, one_idx=None,
        min_improvement: float = 1e-3,
        exact_svd: bool = True
) -> Tuple[Tuple[Tensor,Tensor], float, int]:
    zero_idx = zero_idx or []
    one_idx  = one_idx  or []

    old_obj = compute_objective_value(X_init, Y_init, U, mu, lambda_reg)

    Y = solve_sparse_problem(U - X_init, mu, k_sparse, zero_idx, one_idx)
    X = solve_rank_problem(U - Y, lambda_reg, k_rank, exact_svd)
    new_obj = compute_objective_value(X, Y, U, mu, lambda_reg)

    steps = 1
    while (old_obj - new_obj) / old_obj > min_improvement:
        Y = solve_sparse_problem(U - X, mu, k_sparse, zero_idx, one_idx)
        X = solve_rank_problem(U - Y, lambda_reg, k_rank, exact_svd)
        old_obj, new_obj = new_obj, compute_objective_value(X, Y, U, mu, lambda_reg)
        steps += 1
    return (X, Y), new_obj, steps

# ---------- public API ---------- #

def SLR_AM(
        U: Tensor,
        mu: float, lambda_reg: float,
        k_sparse: int, k_rank: int,
        device: str = "cpu",
        zero_idx: Optional[List[Tuple[int,int]]] = None,
        one_idx:  Optional[List[Tuple[int,int]]] = None,
        random_restarts: int = 1,
        min_improvement: float = 1e-3,
        exact_svd: bool = True
) -> Tuple[Tuple[Tensor,Tensor], float]:
    """
    Alternating minimization with optional multiple random restarts.
    Set device="cuda" (or "cuda:0", etc.) to leverage GPU.
    """
    device = torch.device(device)
    U = _to(U, device)
    zero_idx = zero_idx or []
    one_idx  = one_idx  or []

    m, n = U.shape
    best_obj, best_sol = float("inf"), None

    for _ in range(random_restarts):
        X0 = torch.randn(m, n, device=device)
        X0 = project_matrix(X0, k_rank, exact_svd)
        Y0 = torch.randn(m, n, device=device)
        Y0 = solve_sparse_problem(Y0, mu, k_sparse, zero_idx, one_idx)

        (X, Y), obj, _ = iterate_X_Y(
            U, mu, lambda_reg, k_sparse, k_rank,
            X0, Y0, zero_idx, one_idx,
            min_improvement, exact_svd
        )
        if obj < best_obj:
            best_obj, best_sol = obj, (X, Y)
    return best_sol, best_obj

