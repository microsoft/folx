from typing import Callable, Protocol

from .api import Array


__all__ = [
    "Laplacian",
    "LaplacianOperator",
]


class Laplacian(Protocol):
    def __call__(self, x: Array) -> tuple[Array, Array]:
        ...


class LaplacianOperator(Protocol):
    def __call__(self, f: Callable[[Array], Array]) -> Laplacian:
        ...
