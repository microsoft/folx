import functools
import logging
from typing import Any, Callable, Sequence, TypeVar

import jax

from folx.vmap import batched_vmap

from .memory import compute_memory

F = TypeVar('F', bound=Callable)


def auto_batched_vmap(
    fn: F,
    in_axes: int | None | Sequence[Any] = 0,
    out_axes: Any = 0,
    static_argnums: int | Sequence[int] = (),
    max_fraction: float = 1.0,
) -> F:
    """
    This is an experimental function that will automatically determine the best batch size
    for batched_vmap by attempting to anlyze the memory usage for a single and 2 samples.

    This function is not guaranteed to work in all cases and may be removed in the future.

    Args:
        fn: The function to batch.
        in_axes: The axes to batch over.
        out_axes: The axes to batch over.
        max_fraction: The maximum fraction of memory to use.

    Returns:
        The batched function.
    """
    mem = jax.devices()[0].memory_stats()['bytes_limit']  # type: ignore
    target_mem = mem * max_fraction

    vmapped_fn = functools.partial(batched_vmap, fn, in_axes=in_axes, out_axes=out_axes)

    @functools.wraps(fn)
    def result(*args, **kwargs):
        try:
            single_fn = vmapped_fn(max_batch_size=1)
            pair_fn = vmapped_fn(max_batch_size=2)
            with jax.ensure_compile_time_eval():
                single_cost = compute_memory(
                    single_fn,
                    *args,
                    static_argnums=static_argnums,
                    **kwargs,
                )
                pair_cost = compute_memory(
                    pair_fn,
                    *args,
                    static_argnums=static_argnums,
                    **kwargs,
                )
            element_cost = pair_cost - single_cost
            base_cost = single_cost - element_cost

            if element_cost == 0:
                max_samples = 2147483648
            else:
                max_samples = int((target_mem - base_cost) // element_cost)
                max_samples = max(max_samples, 1)
            return vmapped_fn(max_samples)(*args, **kwargs)
        except Exception as e:
            logging.warn(f'Failed to auto batch {fn.__name__}: {e}')
            logging.warn('Defaulting to batch size 1.')
            return vmapped_fn(max_batch_size=1)(*args, **kwargs)

    return result
