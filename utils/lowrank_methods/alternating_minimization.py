"""
Alternating minimization for sparse plus low rank matrix decomposition.
"""
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
from typing import Tuple, List, Optional, Dict

def compute_objective_value(X: np.ndarray, Y: np.ndarray, U: np.ndarray, 
                          mu: float, lambda_reg: float) -> float:
    """
    Compute the objective value of the optimization problem for solution (X, Y).
    
    The objective is: ||U - X - Y||_F^2 + lambda * ||X||_F^2 + mu * ||Y||_F^2
    
    Args:
        X: Low rank matrix
        Y: Sparse matrix
        U: Input data matrix
        mu: Regularization parameter for sparse matrix penalty
        lambda_reg: Regularization parameter for low rank matrix penalty
        
    Returns:
        The objective value
    """
    return (np.linalg.norm(U - X - Y, 'fro')**2 + 
            lambda_reg * np.linalg.norm(X, 'fro')**2 + 
            mu * np.linalg.norm(Y, 'fro')**2)


def is_feasible(X: np.ndarray, Y: np.ndarray, k_sparse: int, k_rank: int, 
                epsilon: float = 1e-10) -> bool:
    """
    Verify that solution (X, Y) is feasible under constraints.
    
    Args:
        X: Low rank matrix
        Y: Sparse matrix
        k_sparse: Maximum number of non-zero elements in Y
        k_rank: Maximum rank of X
        epsilon: Tolerance for numerical precision
        
    Returns:
        True if constraints are satisfied, False otherwise
    """
    # Verify sparsity constraint
    num_nonzero = np.count_nonzero(np.abs(Y) > epsilon)
    if num_nonzero > k_sparse:
        return False
    
    # Verify rank constraint
    rank_X = np.linalg.matrix_rank(X)
    if rank_X > k_rank:
        return False
    
    return True


def project_matrix(A: np.ndarray, k_rank: int, exact_svd: bool = True) -> np.ndarray:
    """
    Project matrix A onto its first k_rank principal components.
    
    Args:
        A: Input matrix
        k_rank: Desired rank of projection
        exact_svd: If True, compute exact truncated SVD. If False, use randomized SVD
        
    Returns:
        Low rank projection of A
    """
    if k_rank >= min(A.shape):
        return A
        
    if exact_svd:
        try:
            U, s, Vt = svds(A, k=k_rank, which='LM')
            # svds returns in ascending order, we want descending
            idx = np.argsort(s)[::-1]
            U, s, Vt = U[:, idx], s[idx], Vt[idx, :]
        except:
            # Fallback to full SVD if svds fails
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            U, s, Vt = U[:, :k_rank], s[:k_rank], Vt[:k_rank, :]
    else:
        U, s, Vt = randomized_svd(A, n_components=k_rank, random_state=42)
    
    return U @ np.diag(s) @ Vt


def construct_binary_matrix(A: np.ndarray, k_sparse: int,
                          zero_indices: List[Tuple[int, int]] = None,
                          one_indices: List[Tuple[int, int]] = None) -> np.ndarray:
    """
    Construct binary matrix S where S_ij = 1 if A_ij is among k_sparse largest entries.
    
    Args:
        A: Input matrix
        k_sparse: Desired sparsity
        zero_indices: Indices that must be zero
        one_indices: Indices that must be one
        
    Returns:
        Binary sparsity pattern matrix
    """
    if zero_indices is None:
        zero_indices = []
    if one_indices is None:
        one_indices = []
    
    n, m = A.shape
    S = np.zeros((n, m))
    
    # Set forced ones
    for i, j in one_indices:
        S[i, j] = 1
    
    # Calculate remaining sparsity budget
    remaining_sparse = k_sparse - len(one_indices)
    
    if remaining_sparse <= 0:
        return S
    
    # Create mask for available positions (not forced zeros or ones)
    available_mask = np.ones((n, m), dtype=bool)
    for i, j in zero_indices + one_indices:
        available_mask[i, j] = False
    
    # Get absolute values of available entries
    available_values = np.abs(A[available_mask])
    available_positions = np.where(available_mask)
    
    if len(available_values) <= remaining_sparse:
        # Set all available positions
        S[available_positions] = 1
    else:
        # Select top k entries
        threshold_idx = len(available_values) - remaining_sparse
        threshold = np.partition(available_values, threshold_idx)[threshold_idx]
        
        for idx, (i, j) in enumerate(zip(*available_positions)):
            if np.abs(A[i, j]) >= threshold:
                S[i, j] = 1
                remaining_sparse -= 1
                if remaining_sparse == 0:
                    break
    
    return S


def solve_sparse_problem(U_tilde: np.ndarray, mu: float, k_sparse: int,
                        zero_indices: List[Tuple[int, int]] = None,
                        one_indices: List[Tuple[int, int]] = None) -> np.ndarray:
    """
    Solve the sparse subproblem.
    
    Args:
        U_tilde: Input matrix
        mu: Regularization parameter
        k_sparse: Maximum sparsity
        zero_indices: Indices constrained to be zero
        one_indices: Indices constrained to be one
        
    Returns:
        Sparse matrix solution
    """
    S = construct_binary_matrix(U_tilde, k_sparse, zero_indices, one_indices)
    Y = (S * U_tilde) / (1 + mu)
    return Y


def solve_rank_problem(U_tilde: np.ndarray, lambda_reg: float, k_rank: int,
                      exact_svd: bool = True) -> np.ndarray:
    """
    Solve the rank subproblem.
    
    Args:
        U_tilde: Input matrix
        lambda_reg: Regularization parameter
        k_rank: Maximum rank
        exact_svd: Whether to use exact SVD
        
    Returns:
        Low rank matrix solution
    """
    return project_matrix(U_tilde / (1 + lambda_reg), k_rank, exact_svd=exact_svd)


def iterate_X_Y(U: np.ndarray, mu: float, lambda_reg: float, 
                k_sparse: int, k_rank: int, X_init: np.ndarray, Y_init: np.ndarray,
                zero_indices: List[Tuple[int, int]] = None,
                one_indices: List[Tuple[int, int]] = None,
                min_improvement: float = 0.001,
                exact_svd: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], float, int]:
    """
    Compute feasible solution by iteratively solving sparse and rank subproblems.
    
    Args:
        U: Input data matrix
        mu: Regularization parameter for sparse matrix
        lambda_reg: Regularization parameter for low rank matrix
        k_sparse: Maximum sparsity
        k_rank: Maximum rank
        X_init: Initial low rank matrix
        Y_init: Initial sparse matrix
        zero_indices: Indices constrained to be zero
        one_indices: Indices constrained to be one
        min_improvement: Minimum fractional improvement to continue
        exact_svd: Whether to use exact SVD
        
    Returns:
        Tuple containing:
        - (X, Y): Final solution matrices
        - Final objective value
        - Number of iterations
    """
    if zero_indices is None:
        zero_indices = []
    if one_indices is None:
        one_indices = []
    
    old_objective = compute_objective_value(X_init, Y_init, U, mu, lambda_reg)
    
    Y = solve_sparse_problem(U - X_init, mu, k_sparse, zero_indices, one_indices)
    X = solve_rank_problem(U - Y, lambda_reg, k_rank, exact_svd=exact_svd)
    
    new_objective = compute_objective_value(X, Y, U, mu, lambda_reg)
    step_count = 1
    
    # Continue while improvement is significant
    while (old_objective - new_objective) / old_objective > min_improvement:
        Y = solve_sparse_problem(U - X, mu, k_sparse, zero_indices, one_indices)
        X = solve_rank_problem(U - Y, lambda_reg, k_rank, exact_svd=exact_svd)
        
        old_objective = new_objective
        new_objective = compute_objective_value(X, Y, U, mu, lambda_reg)
        step_count += 1
    
    return (X, Y), new_objective, step_count


def SLR_AM(U: np.ndarray, mu: float, lambda_reg: float, k_sparse: int, k_rank: int,
           zero_indices: List[Tuple[int, int]] = None,
           one_indices: List[Tuple[int, int]] = None,
           random_restarts: int = 1,
           min_improvement: float = 0.001,
           exact_svd: bool = True,
           hybrid_svd: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """
    Compute feasible solution using alternating minimization with random restarts.
    
    Args:
        U: Input data matrix
        mu: Regularization parameter for sparse matrix
        lambda_reg: Regularization parameter for low rank matrix
        k_sparse: Maximum sparsity
        k_rank: Maximum rank
        zero_indices: Indices constrained to be zero
        one_indices: Indices constrained to be one
        random_restarts: Number of random restarts
        min_improvement: Minimum fractional improvement to continue
        exact_svd: Whether to use exact SVD
        hybrid_svd: Whether to use hybrid SVD approach
        
    Returns:
        Tuple containing:
        - (X, Y): Best solution matrices found
        - Best objective value achieved
    """
    if zero_indices is None:
        zero_indices = []
    if one_indices is None:
        one_indices = []
    
    # Support rectangular U: get both dimensions
    rows, cols = U.shape
    best_objective = float('inf')
    best_solution = None
    
    for restart in range(random_restarts):
        # Initialize with random feasible solution matching U's shape
        X_init = np.random.randn(rows, cols)
        X_init = project_matrix(X_init, k_rank, exact_svd=exact_svd)
        
        Y_init = np.random.randn(rows, cols)
        Y_init = solve_sparse_problem(Y_init, mu, k_sparse, zero_indices, one_indices)
        
        # Run alternating minimization
        solution, objective, _ = iterate_X_Y(
            U, mu, lambda_reg, k_sparse, k_rank, X_init, Y_init,
            zero_indices, one_indices, min_improvement, exact_svd
        )
        
        # Keep best solution
        if objective < best_objective:
            best_objective = objective
            best_solution = solution
    
    return best_solution, best_objective


def cross_validate_parameters(U: np.ndarray, k_sparse: int, k_rank: int,
                            num_samples: int = 5, train_frac: float = 0.8,
                            candidate_lambdas: List[float] = None,
                            candidate_mus: List[float] = None,
                            exact_svd: bool = True,
                            hybrid_svd: bool = False) -> Tuple[float, float, Dict]:
    """
    Perform cross validation to select regularization parameters.
    
    Args:
        U: Input data matrix
        k_sparse: Maximum sparsity
        k_rank: Maximum rank
        num_samples: Number of cross validation samples
        train_frac: Fraction of data for training
        candidate_lambdas: Lambda values to test
        candidate_mus: Mu values to test
        exact_svd: Whether to use exact SVD
        hybrid_svd: Whether to use hybrid SVD
        
    Returns:
        Tuple containing:
        - Best lambda value
        - Best mu value
        - Dictionary of parameter scores
    """
    if candidate_lambdas is None:
        candidate_lambdas = [0.01, 0.1, 1.0, 10.0]
    if candidate_mus is None:
        candidate_mus = [0.01, 0.1, 1.0, 10.0]
    
    n = U.shape[0]
    val_dim = int(np.floor(n * (1 - np.sqrt(train_frac))))
    train_dim = n - val_dim
    
    param_scores = {}
    for lambda_mult in candidate_lambdas:
        for mu_mult in candidate_mus:
            param_scores[(lambda_mult, mu_mult)] = 0
    
    for trial in range(num_samples):
        # Random permutation for train/validation split
        permutation = np.random.permutation(n)
        val_indices = permutation[:val_dim]
        train_indices = permutation[val_dim:]
        
        val_data = U[np.ix_(val_indices, val_indices)]
        train_data = U[np.ix_(train_indices, train_indices)]
        
        LL_block_data = U[np.ix_(val_indices, train_indices)]
        UR_block_data = U[np.ix_(train_indices, val_indices)]
        
        for lambda_mult in candidate_lambdas:
            for mu_mult in candidate_mus:
                this_lambda = lambda_mult / np.sqrt(n)
                this_mu = mu_mult / np.sqrt(n)
                
                sol, _ = SLR_AM(train_data, this_mu, this_lambda, k_sparse, k_rank,
                              exact_svd=exact_svd)
                
                # Compute validation error
                X_sol, _ = sol
                try:
                    val_estimate = LL_block_data @ np.linalg.pinv(X_sol) @ UR_block_data
                    val_error = (np.linalg.norm(val_estimate - val_data, 'fro')**2 / 
                               np.linalg.norm(val_data, 'fro')**2)
                except:
                    val_error = 1e9  # Large penalty for numerical issues
                
                param_scores[(lambda_mult, mu_mult)] += val_error / num_samples
    
    # Find best parameters
    best_score = float('inf')
    best_params = None
    for param, score in param_scores.items():
        if score < best_score:
            best_score = score
            best_params = param
    
    return best_params[0], best_params[1], param_scores
