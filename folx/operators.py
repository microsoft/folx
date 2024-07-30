from dataclasses import dataclass
from typing import Callable, Protocol

import jax
import jax.numpy as jnp

from .api import Array
from .interpreter import forward_laplacian

__all__ = [
    'Laplacian',
    'LaplacianOperator',
    'ForwardLaplacianOperator',
    'LoopLaplacianOperator',
    'ParallelLaplacianOperator',
]


class Laplacian(Protocol):
    def __call__(self, x: Array) -> tuple[Array, Array]: ...


class LaplacianOperator(Protocol):
    def __call__(self, f: Callable[[Array], Array]) -> Laplacian: ...


@dataclass(frozen=True)
class ForwardLaplacianOperator(LaplacianOperator):
    sparsity_threshold: float | int

    def __call__(self, f):
        fwd_lapl = forward_laplacian(f, self.sparsity_threshold)

        def lap(x: Array):
            result = fwd_lapl(x)
            return result.laplacian, result.jacobian.dense_array

        return lap


class LoopLaplacianOperator(LaplacianOperator):
    @staticmethod
    def __call__(f):
        @jax.jit
        def laplacian(x: jax.Array):
            x_shape = x.shape
            x = x.reshape(-1)
            n = x.shape[0]
            eye = jnp.eye(n, dtype=x.dtype)

            def f_(x):
                return f(x.reshape(x_shape))

            grad_f = jax.grad(f_)
            jacobian, dgrad_f = jax.linearize(grad_f, x)

            _, laplacian = jax.lax.scan(
                lambda i, _: (i + 1, dgrad_f(eye[i])[i]), 0, None, length=n
            )
            return laplacian, jacobian

        return laplacian


class ParallelLaplacianOperator(LaplacianOperator):
    @staticmethod
    def __call__(f):
        @jax.jit
        def laplacian(x: jax.Array):
            x = x.reshape(-1)
            n = x.shape[0]
            eye = jnp.eye(n, dtype=x.dtype)
            grad_f = jax.grad(f)
            jacobian, dgrad_f = jax.linearize(grad_f, x)

            laplacian = jnp.diagonal(jax.vmap(dgrad_f)(eye))
            return laplacian, jacobian

        return laplacian
