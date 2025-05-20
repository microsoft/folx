from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def sum_columns(x: jax.Array) -> jax.Array:
    return x.sum(axis=1, keepdims=True)


def get_value_or_laplacian_block_spec(
    seq_len: int,
    head_len: int,
    seq_block_len: Optional[int] = None,
    is_q_like: bool = False,
):
    r"""Return block specification for arrays (or their laplacians) involved in MHA.

    These arrays are of shape ``(batch_len, seq_len, num_heads, head_len)``.

    Args:
        seq_len (int): The sequence length.
        head_len (int): The head length.
        seq_block_len (int, optional): If ``None``, there is no blocking of the sequence
            dimension. Otherwise, the sequence dimension is blocked into blocks of length
            ``seq_block_len``. Defaults to ``None``.
        is_q_like (bool, optional): Whether the grid indexing should vary over the
            sequence axis (like q) or not (like v)
    """
    if seq_block_len is None:
        return pl.BlockSpec(
            index_map=lambda i, j: (i, 0, j, 0),
            block_shape=(None, seq_len, None, head_len),
        )
    elif is_q_like:
        return pl.BlockSpec(
            index_map=lambda i, j, k: (i, j, k, 0),
            block_shape=(None, seq_block_len, None, head_len),
        )
    else:
        return pl.BlockSpec(
            index_map=lambda i, _j, k: (i, 0, k, 0),
            block_shape=(None, seq_block_len, None, head_len),
        )


def get_jacobian_block_spec(
    input_len: int,
    seq_len: int,
    head_len: int,
    seq_block_len: Optional[int] = None,
    is_q_like: bool = False,
):
    r"""Return block specification for jacobians of arrays involved in MHA.

    These arrays are of shape ``(input_len, batch_len, seq_len, num_heads, head_len)``.

    Args:
        input_len (int): The length of the original input to the model.
        seq_len (int): The sequence length.
        head_len (int): The head length.
        seq_block_len (int, optional): If ``None``, there is no blocking of the sequence
            dimension. Otherwise, the sequence dimension is blocked into blocks of length
            ``seq_block_len``. Defaults to ``None``.
        is_q_like (bool, optional): Whether the grid indexing should vary over the
            sequence axis (like q) or not (like v)
    """
    if seq_block_len is None:
        return pl.BlockSpec(
            index_map=lambda i, j: (0, i, 0, j, 0),
            block_shape=(input_len, None, seq_len, None, head_len),
        )
    elif is_q_like:
        return pl.BlockSpec(
            index_map=lambda i, j, k: (0, i, j, k, 0),
            block_shape=(input_len, None, seq_block_len, None, head_len),
        )
    else:
        return pl.BlockSpec(
            index_map=lambda i, _j, k: (0, i, 0, k, 0),
            block_shape=(input_len, None, seq_block_len, None, head_len),
        )


def get_mask_block_spec(seq_len: int, seq_block_len: Optional[int] = None):
    r"""Return block specification for the sequence mask used in MHA.

    This mask is of shape ``(batch_len, seq_len)``.

    Args:
        seq_len (int): The sequence length.
        seq_block_len (int, optional): If ``None``, there is no blocking of the sequence
            dimension. Otherwise, the sequence dimension is blocked into blocks of length
            ``seq_block_len``. In this function, this is used only to determine the number
            of dimensions of the grid. Defaults to ``None``.
    """
    if seq_block_len is None:
        return pl.BlockSpec(index_map=lambda i, _j: (i, 0), block_shape=(None, seq_len))
    return pl.BlockSpec(index_map=lambda i, _j, _k: (i, 0), block_shape=(None, seq_len))


def get_input_mask_block_spec(input_len: int, seq_block_len: Optional[int] = None):
    r"""Return block specification for the input mask used in MHA.

    This mask is of shape ``(input_len, batch_len)``.

    Args:
        seq_len (int): The sequence length.
        seq_block_len (int, optional): If ``None``, there is no blocking of the sequence
            dimension. Otherwise, the sequence dimension is blocked into blocks of length
            ``seq_block_len``. In this function, this is used only to determine the number
            of dimensions of the grid. Defaults to ``None``.
    """
    if seq_block_len is None:
        return pl.BlockSpec(
            index_map=lambda i, _j: (0, i), block_shape=(input_len, None)
        )
    return pl.BlockSpec(
        index_map=lambda i, _j, _k: (0, i), block_shape=(input_len, None)
    )


def get_lse_block_spec(
    seq_len: int,
    seq_block_len: Optional[int] = None,
    is_q_blocking: bool = True,
) -> pl.BlockSpec:
    r"""Return block specification for the auxuliary logsumexp output of the MHSEA kernel.

    This array is of shape ``(batch_len, seq_len, num_heads)``.

    Args:
        seq_len (int): The sequence length.
        head_len (int): The head length.
        seq_block_len (int, optional): If ``None``, there is no blocking of the sequence
            dimension. Otherwise, the sequence dimension is blocked into blocks of length
            ``seq_block_len``. Defaults to ``None``.
        is_q_blocking (bool, optional): Whether the grid indexing should vary over the
            sequence axis (like q) or not (like v)
    """
    if seq_block_len is None:
        return pl.BlockSpec(
            index_map=lambda i, j: (i, 0, j), block_shape=(None, seq_len, None)
        )
    elif is_q_blocking:
        return pl.BlockSpec(
            index_map=lambda i, j, k: (i, j, k), block_shape=(None, seq_block_len, None)
        )
    else:
        return pl.BlockSpec(
            index_map=lambda i, _j, k: (i, _j, k), block_shape=(None, seq_len, None)
        )


def create_grid(
    batch_len: int, seq_len: int, num_heads: int, q_block_len: int | None
) -> Tuple[int, int] | Tuple[int, int, int]:
    """Helper method to create pallas grids.

    When `q_block_len` is `None`, this creates a grid over the batch
    and head dimensions. When, `q_block_len` is an integer, this creates
    a grid over the batch, head and the different slices of `q`.

    Args:
        batch_len (int): Batch dimension.
        seq_len (int): The sequence length.
        num_heads (int): The number of heads.
        q_block_len (int | None): The length of each block of queries, or None if
            q-blocking is disabled. Blocking is implemented over the sequence
            dimension.
    """
    return (
        (batch_len, num_heads)
        if q_block_len is None
        else (batch_len, seq_len // q_block_len, num_heads)
    )


def compute_q_and_kv_block_len(
    seq_len: int, q_block_len: int | None
) -> Tuple[int | None, int | None]:
    """Helper method to compute block lengths.

    Args:
        seq_len (int): The sequence length.
        q_block_len (int | None): The length of each block of queries, or None if
            q-blocking is disabled. Blocking is implemented over the sequence
            dimension.
    Returns:
        q_block_len (int): The length of q-blocks, promoted to an integer even
            when the input is `None`
        kv_block_len (int): The length of k and v blocks, we currently do
            not implement blocking over k and v.
    """
    if q_block_len is not None:
        q_block_len = min(seq_len, q_block_len)
    kv_block_len = None if q_block_len is None else seq_len
    return q_block_len, kv_block_len


def big_number(dtype) -> float:
    if dtype == jnp.float16:
        return 1e10
    elif dtype == jnp.bfloat16:
        return 1e20
    elif dtype == jnp.float32:
        return 1e20
    elif dtype == jnp.float64:
        return 1e40
    else:
        raise ValueError(f'Unexpected dtype {dtype}')
