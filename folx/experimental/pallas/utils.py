from typing import Optional, Tuple

import jax
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
        return pl.BlockSpec(lambda i, j: (i, 0, j, 0), (None, seq_len, None, head_len))
    elif is_q_like:
        return pl.BlockSpec(lambda i, j, k: (i, j, k, 0), (None, seq_block_len, None, head_len))
    else:
        return pl.BlockSpec(lambda i, _j, k: (i, 0, k, 0), (None, seq_block_len, None, head_len))


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
            lambda i, j: (0, i, 0, j, 0), (input_len, None, seq_len, None, head_len)
        )
    elif is_q_like:
        return pl.BlockSpec(
            lambda i, j, k: (0, i, j, k, 0), (input_len, None, seq_block_len, None, head_len)
        )
    else:
        return pl.BlockSpec(
            lambda i, _j, k: (0, i, 0, k, 0), (input_len, None, seq_block_len, None, head_len)
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
        return pl.BlockSpec(lambda i, _j: (i, 0), (None, seq_len))
    return pl.BlockSpec(lambda i, _j, _k: (i, 0), (None, seq_len))


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
        return pl.BlockSpec(lambda i, _j: (0, i), (input_len, None))
    return pl.BlockSpec(lambda i, _j, _k: (0, i), (input_len, None))


def create_grid(
    batch_len: int, seq_len: int, num_heads: int, q_block_len: int | None
) -> Tuple[int, int] | Tuple[int, int, int]:
    return (
        (batch_len, num_heads)
        if q_block_len is None
        else (batch_len, seq_len // q_block_len, num_heads)
    )


def compute_q_and_kv_block_len(
    seq_len: int, q_block_len: int | None
) -> Tuple[int | None, int | None]:
    if q_block_len is not None:
        q_block_len = min(seq_len, q_block_len)
    kv_block_len = None if q_block_len is None else seq_len
    return q_block_len, kv_block_len
