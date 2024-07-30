import importlib.metadata

from .interpreter import forward_laplacian
from .operators import (
    ForwardLaplacianOperator,
    LaplacianOperator,
    LoopLaplacianOperator,
    ParallelLaplacianOperator,
)
from .vmap import batched_vmap
from .wrapped_functions import deregister_function, register_function
from .wrapper import warp_without_fwd_laplacian, wrap_forward_laplacian

__all__ = [
    'batched_vmap',
    'forward_laplacian',
    'ForwardLaplacianOperator',
    'LaplacianOperator',
    'LoopLaplacianOperator',
    'ParallelLaplacianOperator',
    'wrap_forward_laplacian',
    'warp_without_fwd_laplacian',
    'deregister_function',
    'register_function',
]

__version__ = importlib.metadata.version(__package__ or __name__)
