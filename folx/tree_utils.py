from typing import Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PyTree

T = TypeVar('T', bound=PyTree[ArrayLike])


def tree_scale(tree: T, x: ArrayLike) -> T:
    return jtu.tree_map(lambda a: a * x, tree)


def tree_mul(tree: T, x: T | ArrayLike) -> T:
    if isinstance(x, ArrayLike):
        return tree_scale(tree, x)
    return jtu.tree_map(lambda a, b: a * b, tree, x)


def tree_shift(tree1: T, x: ArrayLike) -> T:
    return jtu.tree_map(lambda a: a + x, tree1)


def tree_add(tree1: T, tree2: T | ArrayLike) -> T:
    if isinstance(tree2, ArrayLike):
        return tree_shift(tree1, tree2)
    return jtu.tree_map(lambda a, b: a + b, tree1, tree2)


def tree_sub(tree1: T, tree2: T) -> T:
    return jtu.tree_map(lambda a, b: a - b, tree1, tree2)


def tree_dot(a: T, b: T) -> Array:
    return jtu.tree_reduce(
        jnp.add, jtu.tree_map(jnp.sum, jax.tree_map(jax.lax.mul, a, b))
    )


def tree_sum(tree: PyTree[ArrayLike]) -> Array:
    return jtu.tree_reduce(jnp.add, jtu.tree_map(jnp.sum, tree))


def tree_squared_norm(tree: PyTree[ArrayLike]) -> Array:
    return jtu.tree_reduce(
        jnp.add, jtu.tree_map(lambda x: jnp.einsum('...,...->', x, x), tree)
    )


def tree_concat(trees: Sequence[T], axis: int = 0) -> T:
    return jtu.tree_map(lambda *args: jnp.concatenate(args, axis=axis), *trees)


def tree_split(tree: PyTree[Array], sizes: tuple[int]) -> tuple[PyTree[Array], ...]:
    idx = 0
    result: list[PyTree[Array]] = []
    for s in sizes:
        result.append(jtu.tree_map(lambda x: x[idx : idx + s], tree))
        idx += s
    result.append(jtu.tree_map(lambda x: x[idx:], tree))
    return tuple(result)


def tree_idx(tree: T, idx) -> T:
    return jtu.tree_map(lambda x: x[idx], tree)


def tree_expand(tree: T, axis) -> T:
    return jtu.tree_map(lambda x: jnp.expand_dims(x, axis), tree)


def tree_take(tree: T, idx, axis) -> T:
    def take(x):
        indices = idx
        if isinstance(indices, slice):
            slices = [slice(None)] * x.ndim
            slices[axis] = idx
            return x[tuple(slices)]
        return jnp.take(x, indices, axis)

    return jtu.tree_map(take, tree)
