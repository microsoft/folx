from typing import Any, Callable, Protocol

import jax
import numpy as np

ArrayLike = jax.Array | float | int | bool | np.ndarray | np.number | np.bool_
Array = jax.Array
PyTree = Any

class Laplacian(Protocol):
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        ...


class LaplacianOperator(Protocol):
    def __call__(self, f: Callable[[jax.Array], jax.Array]) -> Laplacian:
        ...
