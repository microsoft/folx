import functools
from typing import Any, Callable, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

F = TypeVar('F', bound=Callable)


def batched_vmap(
    fn: F,
    max_batch_size: int,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
) -> F:
    if isinstance(in_axes, list):
        in_axes = tuple(in_axes)

    @functools.wraps(fn)
    def result(*args, **kwargs):
        wrapped_fn = functools.partial(fn, **kwargs)

        in_axes_flat, in_tree = jtu.tree_flatten(
            in_axes, lambda x: isinstance(x, int) or x is None
        )
        in_args = in_tree.flatten_up_to(args)

        mask = [a is not None for a in in_axes_flat]
        mapped_args = [a for a, m in zip(in_args, mask) if m]
        mapped_axes = [a for a, m in zip(in_axes_flat, mask) if m]
        static_args = [a for a, m in zip(in_args, mask) if not m]

        def merge(mapped_args):
            x_iter = iter(mapped_args)
            y_iter = iter(static_args)
            return tuple(next(x_iter) if m else next(y_iter) for m in mask)

        batch_size = jtu.tree_leaves(mapped_args[0])[0].shape[mapped_axes[0]]
        assert (
            batch_size == x.shape[ax]
            for arg, ax in zip(mapped_args, mapped_axes)
            for x in jtu.tree_leaves(arg)
        )
        if batch_size <= max_batch_size:
            return jax.vmap(wrapped_fn, in_axes=in_axes, out_axes=out_axes)(*args)

        # Split into batches.
        num_batches = batch_size // max_batch_size
        remainder = batch_size % max_batch_size
        leading_args = [
            jtu.tree_map(lambda x: jnp.moveaxis(x, ax, 0), arg)
            for arg, ax in zip(mapped_args, mapped_axes)
        ]
        loop_args = jtu.tree_map(
            lambda x: x[remainder:].reshape(num_batches, max_batch_size, *x.shape[1:]),
            leading_args,
        )

        vmapped_fn = jax.vmap(
            lambda x: wrapped_fn(*in_tree.unflatten(merge(x))), in_axes=0, out_axes=0
        )

        def inner(carry, x):
            return carry, vmapped_fn(x)

        result = jax.lax.scan(inner, None, loop_args, length=num_batches)[1]
        result = jtu.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), result)
        if remainder > 0:
            remainder_args = jtu.tree_map(lambda x: x[:remainder], leading_args)
            remainder_result = vmapped_fn(remainder_args)
            result = jtu.tree_map(
                lambda x, r: jnp.concatenate([r, x], axis=0), result, remainder_result
            )

        out_axes_flat, out_tree = jtu.tree_flatten(out_axes)
        result = tuple(
            jtu.tree_map(lambda x: jnp.moveaxis(x, 0, ax), r)
            for r, ax in zip(out_tree.flatten_up_to(result), out_axes_flat)
        )
        return out_tree.unflatten(result)

    return result
