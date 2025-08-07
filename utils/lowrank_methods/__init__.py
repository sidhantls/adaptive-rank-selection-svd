"""
Benchmark methods for sparse plus low rank matrix decomposition.
"""

from .spcp import stable_principal_component_pursuit, spcp_with_rank_sparsity_constraints
from .alt_proj import alternating_projection, alternating_projection_with_sparsity, robust_alternating_projection
from .scaled_gd import scaled_gradient_descent, momentum_scaled_gd, adaptive_scaled_gd
from .fast_rpca import fast_rpca_alm, fast_rpca_ialm, fast_rpca_with_constraints, accelerated_rpca

__all__ = [
    'stable_principal_component_pursuit',
    'spcp_with_rank_sparsity_constraints', 
    'alternating_projection',
    'alternating_projection_with_sparsity',
    'robust_alternating_projection',
    'scaled_gradient_descent',
    'momentum_scaled_gd',
    'adaptive_scaled_gd',
    'fast_rpca_alm',
    'fast_rpca_ialm', 
    'fast_rpca_with_constraints',
    'accelerated_rpca'
]
