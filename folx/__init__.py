from dataclasses import dataclass

from .api import Array
from .interpreter import forward_laplacian
from .types import LaplacianOperator
from .vmap import batched_vmap
from .wrapper import wrap_forward_laplacian, warp_without_fwd_laplacian
from .wrapped_functions import deregister_function, register_function



@dataclass(frozen=True)
class ForwardLaplacianOperator(LaplacianOperator):
    sparsity_threshold: float | int

    def __call__(self, f):
        fwd_lapl = forward_laplacian(f, self.sparsity_threshold)

        def lap(x: Array):
            result = fwd_lapl(x)
            return result.laplacian, result.jacobian.dense_array

        return lap


__all__ = [
    'batched_vmap',
    'forward_laplacian',
    'ForwardLaplacianOperator',
    'wrap_forward_laplacian',
    'warp_without_fwd_laplacian',
    'deregister_function',
    'register_function',
]
