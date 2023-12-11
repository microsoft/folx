from typing import Callable, Protocol

import jax


class Laplacian(Protocol):
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        ...


class LaplacianOperator(Protocol):
    def __call__(self, f: Callable[[jax.Array], jax.Array]) -> Laplacian:
        ...
