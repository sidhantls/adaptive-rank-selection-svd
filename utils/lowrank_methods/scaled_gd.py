"""
Scaled Gradient Descent method for sparse plus low rank matrix decomposition.
"""
import numpy as np
from typing import Tuple, Optional


def scaled_gradient_descent(A: np.ndarray, k_rank: int, k_sparse: int,
                          lambda_reg: float = 0.1, mu_reg: float = 0.1,
                          learning_rate: float = 0.01,
                          max_iterations: int = 1000,
                          epsilon: float = 1e-6,
                          line_search: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """
    Scaled gradient descent for sparse plus low rank decomposition.
    
    Solves:
    minimize ||A - X - Y||_F^2 + λ||X||_F^2 + μ||Y||_F^2
    subject to rank(X) ≤ k_rank, ||Y||_0 ≤ k_sparse
    
    Args:
        A: Input matrix
        k_rank: Maximum rank constraint
        k_sparse: Maximum sparsity constraint
        lambda_reg: Regularization parameter for low rank matrix
        mu_reg: Regularization parameter for sparse matrix
        learning_rate: Initial learning rate
        max_iterations: Maximum number of iterations
        epsilon: Convergence tolerance
        line_search: Whether to use line search for step size
        
    Returns:
        Tuple containing:
        - (X, Y): Low rank and sparse matrix decomposition
        - Final objective value
    """
    n = A.shape[0]
    
    # Initialize with feasible solution
    X = np.random.randn(n, n) * 0.1
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s[k_rank:] = 0
    X = U @ np.diag(s) @ Vt
    
    Y = np.random.randn(n, n) * 0.1
    Y_flat = Y.flatten()
    indices = np.argsort(np.abs(Y_flat))[::-1]
    Y_flat[indices[k_sparse:]] = 0
    Y = Y_flat.reshape((n, n))
    
    # Track objective history
    objective_history = []
    current_lr = learning_rate
    
    for iteration in range(max_iterations):
        # Compute current objective
        reconstruction_error = A - X - Y
        objective = (np.linalg.norm(reconstruction_error, 'fro')**2 + 
                    lambda_reg * np.linalg.norm(X, 'fro')**2 + 
                    mu_reg * np.linalg.norm(Y, 'fro')**2)
        objective_history.append(objective)
        
        # Compute gradients
        grad_X = -2 * reconstruction_error + 2 * lambda_reg * X
        grad_Y = -2 * reconstruction_error + 2 * mu_reg * Y
        
        # Store previous iterates for line search
        X_prev, Y_prev = X.copy(), Y.copy()
        
        # Gradient step with scaling
        if line_search and iteration > 0:
            # Simple backtracking line search
            step_size = current_lr
            for _ in range(5):  # Maximum 5 backtracking steps
                X_candidate = X_prev - step_size * grad_X
                Y_candidate = Y_prev - step_size * grad_Y
                
                # Project to feasible set
                X_candidate = project_to_rank(X_candidate, k_rank)
                Y_candidate = project_to_sparsity(Y_candidate, k_sparse)
                
                # Check if objective improved
                reconstruction_new = A - X_candidate - Y_candidate
                objective_new = (np.linalg.norm(reconstruction_new, 'fro')**2 + 
                               lambda_reg * np.linalg.norm(X_candidate, 'fro')**2 + 
                               mu_reg * np.linalg.norm(Y_candidate, 'fro')**2)
                
                if objective_new < objective:
                    X, Y = X_candidate, Y_candidate
                    current_lr = min(step_size * 1.1, learning_rate * 10)
                    break
                else:
                    step_size *= 0.5
            else:
                # If line search failed, use smaller step
                X = X_prev - current_lr * 0.1 * grad_X
                Y = Y_prev - current_lr * 0.1 * grad_Y
                current_lr *= 0.9
        else:
            # Standard gradient step
            X = X - current_lr * grad_X
            Y = Y - current_lr * grad_Y
        
        # Project to feasible set
        X = project_to_rank(X, k_rank)
        Y = project_to_sparsity(Y, k_sparse)
        
        # Check convergence
        if iteration > 10:
            recent_objectives = objective_history[-10:]
            if (max(recent_objectives) - min(recent_objectives)) / max(recent_objectives) < epsilon:
                break
    
    # Final objective computation
    reconstruction_error = A - X - Y
    final_objective = (np.linalg.norm(reconstruction_error, 'fro')**2 + 
                      lambda_reg * np.linalg.norm(X, 'fro')**2 + 
                      mu_reg * np.linalg.norm(Y, 'fro')**2)
    
    return (X, Y), final_objective


def project_to_rank(X: np.ndarray, k_rank: int) -> np.ndarray:
    """Project matrix to rank-k constraint."""
    if k_rank >= min(X.shape):
        return X
    
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s[k_rank:] = 0
    return U @ np.diag(s) @ Vt


def project_to_sparsity(Y: np.ndarray, k_sparse: int) -> np.ndarray:
    """Project matrix to k-sparse constraint."""
    Y_flat = Y.flatten()
    if len(Y_flat) <= k_sparse:
        return Y
    
    indices = np.argsort(np.abs(Y_flat))[::-1]
    Y_proj_flat = np.zeros_like(Y_flat)
    Y_proj_flat[indices[:k_sparse]] = Y_flat[indices[:k_sparse]]
    return Y_proj_flat.reshape(Y.shape)


def momentum_scaled_gd(A: np.ndarray, k_rank: int, k_sparse: int,
                      lambda_reg: float = 0.1, mu_reg: float = 0.1,
                      learning_rate: float = 0.01,
                      momentum: float = 0.9,
                      max_iterations: int = 1000,
                      epsilon: float = 1e-6) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """
    Scaled gradient descent with momentum for sparse plus low rank decomposition.
    
    Args:
        A: Input matrix
        k_rank: Maximum rank constraint
        k_sparse: Maximum sparsity constraint
        lambda_reg: Regularization parameter for low rank matrix
        mu_reg: Regularization parameter for sparse matrix
        learning_rate: Learning rate
        momentum: Momentum parameter
        max_iterations: Maximum iterations
        epsilon: Convergence tolerance
        
    Returns:
        Tuple containing:
        - (X, Y): Low rank and sparse matrix decomposition
        - Final objective value
    """
    n = A.shape[0]
    
    # Initialize with feasible solution
    X = np.random.randn(n, n) * 0.1
    X = project_to_rank(X, k_rank)
    
    Y = np.random.randn(n, n) * 0.1
    Y = project_to_sparsity(Y, k_sparse)
    
    # Initialize momentum terms
    v_X = np.zeros_like(X)
    v_Y = np.zeros_like(Y)
    
    objective_history = []
    
    for iteration in range(max_iterations):
        # Compute current objective
        reconstruction_error = A - X - Y
        objective = (np.linalg.norm(reconstruction_error, 'fro')**2 + 
                    lambda_reg * np.linalg.norm(X, 'fro')**2 + 
                    mu_reg * np.linalg.norm(Y, 'fro')**2)
        objective_history.append(objective)
        
        # Compute gradients
        grad_X = -2 * reconstruction_error + 2 * lambda_reg * X
        grad_Y = -2 * reconstruction_error + 2 * mu_reg * Y
        
        # Update momentum
        v_X = momentum * v_X - learning_rate * grad_X
        v_Y = momentum * v_Y - learning_rate * grad_Y
        
        # Update variables
        X = X + v_X
        Y = Y + v_Y
        
        # Project to feasible set
        X = project_to_rank(X, k_rank)
        Y = project_to_sparsity(Y, k_sparse)
        
        # Check convergence
        if iteration > 10:
            recent_objectives = objective_history[-10:]
            if (max(recent_objectives) - min(recent_objectives)) / max(recent_objectives) < epsilon:
                break
    
    # Final objective
    reconstruction_error = A - X - Y
    final_objective = (np.linalg.norm(reconstruction_error, 'fro')**2 + 
                      lambda_reg * np.linalg.norm(X, 'fro')**2 + 
                      mu_reg * np.linalg.norm(Y, 'fro')**2)
    
    return (X, Y), final_objective


def adaptive_scaled_gd(A: np.ndarray, k_rank: int, k_sparse: int,
                      lambda_reg: float = 0.1, mu_reg: float = 0.1,
                      initial_lr: float = 0.01,
                      max_iterations: int = 1000,
                      epsilon: float = 1e-6) -> Tuple[Tuple[np.ndarray, np.ndarray], float]:
    """
    Adaptive scaled gradient descent with automatic learning rate adjustment.
    
    Args:
        A: Input matrix
        k_rank: Maximum rank constraint
        k_sparse: Maximum sparsity constraint
        lambda_reg: Regularization parameter for low rank matrix
        mu_reg: Regularization parameter for sparse matrix
        initial_lr: Initial learning rate
        max_iterations: Maximum iterations
        epsilon: Convergence tolerance
        
    Returns:
        Tuple containing:
        - (X, Y): Low rank and sparse matrix decomposition
        - Final objective value
    """
    n = A.shape[0]
    
    # Initialize
    X = project_to_rank(np.random.randn(n, n) * 0.1, k_rank)
    Y = project_to_sparsity(np.random.randn(n, n) * 0.1, k_sparse)
    
    learning_rate = initial_lr
    objective_history = []
    
    for iteration in range(max_iterations):
        # Current objective
        reconstruction_error = A - X - Y
        objective = (np.linalg.norm(reconstruction_error, 'fro')**2 + 
                    lambda_reg * np.linalg.norm(X, 'fro')**2 + 
                    mu_reg * np.linalg.norm(Y, 'fro')**2)
        objective_history.append(objective)
        
        # Gradients
        grad_X = -2 * reconstruction_error + 2 * lambda_reg * X
        grad_Y = -2 * reconstruction_error + 2 * mu_reg * Y
        
        # Adaptive learning rate
        if iteration > 0:
            if objective > objective_history[-2]:
                learning_rate *= 0.8  # Decrease if objective increased
            else:
                learning_rate *= 1.02  # Slight increase if objective decreased
            learning_rate = np.clip(learning_rate, 1e-6, 1.0)
        
        # Update
        X_new = X - learning_rate * grad_X
        Y_new = Y - learning_rate * grad_Y
        
        # Project
        X = project_to_rank(X_new, k_rank)
        Y = project_to_sparsity(Y_new, k_sparse)
        
        # Convergence check
        if iteration > 10:
            recent_objectives = objective_history[-5:]
            if (max(recent_objectives) - min(recent_objectives)) / max(recent_objectives) < epsilon:
                break
    
    reconstruction_error = A - X - Y
    final_objective = (np.linalg.norm(reconstruction_error, 'fro')**2 + 
                      lambda_reg * np.linalg.norm(X, 'fro')**2 + 
                      mu_reg * np.linalg.norm(Y, 'fro')**2)
    
    return (X, Y), final_objective
