from typing import Callable, Protocol

import jax
from jaxtyping import Array, ArrayLike, PyTree

__all__ = [
    "Array",
    "ArrayLike",
    "PyTree",
    "Laplacian",
    "LaplacianOperator",
]

class Laplacian(Protocol):
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        ...


class LaplacianOperator(Protocol):
    def __call__(self, f: Callable[[jax.Array], jax.Array]) -> Laplacian:
        ...
