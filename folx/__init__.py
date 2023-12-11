import jax
import jax_dataclasses as jdc

from .interpreter import forward_laplacian
from .types import LaplacianOperator


@jdc.pytree_dataclass
class ForwardLaplacianOperator(LaplacianOperator):
    sparsity_threshold: float = 0.75

    def __call__(self, f):
        fwd_lapl = forward_laplacian(f)

        def lap(x: jax.Array):
            result = fwd_lapl(x)
            return result.laplacian, result.jacobian.dense_array

        return lap


__all__ = ["forward_laplacian", "ForwardLaplacianOperator"]
