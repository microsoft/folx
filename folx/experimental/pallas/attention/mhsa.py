import logging
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .utils import (
    big_number,
    compute_q_and_kv_block_len,
    create_grid,
    get_mask_block_spec,
    get_value_or_laplacian_block_spec,
)


def mhsa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    input_mask: jax.Array,
    kernel: Literal['pallas', 'reference'] = 'pallas',
    interpret: bool = False,
    q_block_len: int | None = None,
    num_warps: int = 2,
    num_stages: int = 2,
) -> jax.Array:
    r"""Pallas implementation of masked multi-head attention.

    Note: the dimensions of the tensor inputs to this function must have dimensions that are
    powers of 2, and any dimension that will participate in a matrix multiplication must
    have dimension at least 16.

    By default, when using pallas, we will run the operation in parallel over the batch and head
    dimensions of the inputs. This is implemented by creating a pallas grid of the relevant size
    and distributing the necessary submatrices to each streaming multiprocessor (SM).
    At some point, the sequence length may become too large to run the entire computation for
    one head on a single SM. In this case, by changing `q_block_len`, we distribute different
    blocks of queries to different SMs.
    """
    del input_mask  # Only used in the forward Laplacian
    batch_len, seq_len, num_heads, head_len = q.shape
    q_block_len, kv_block_len = compute_q_and_kv_block_len(seq_len, q_block_len)

    if kernel == 'pallas':
        kernel_fn = pl.pallas_call(
            partial(mhsa_kernel, q_block_len=q_block_len),
            grid=create_grid(batch_len, seq_len, num_heads, q_block_len),
            in_specs=[
                get_value_or_laplacian_block_spec(seq_len, head_len, q_block_len),
                get_value_or_laplacian_block_spec(seq_len, head_len, kv_block_len),
                get_value_or_laplacian_block_spec(seq_len, head_len, kv_block_len),
                get_mask_block_spec(seq_len, q_block_len),
            ],
            out_specs=get_value_or_laplacian_block_spec(seq_len, head_len, q_block_len),
            out_shape=jax.ShapeDtypeStruct(
                shape=(batch_len, seq_len, num_heads, head_len), dtype=q.dtype
            ),
            compiler_params=dict(
                triton=dict(num_warps=num_warps, num_stages=num_stages)
            ),
            debug=False,
            interpret=interpret,
            name='mhsa',
        )
    elif kernel == 'reference':
        logging.warning(
            'Passing kernel="reference" to function mhsa is not recommended in production, '
            'as it is very slow. Use kernel="pallas" instead.'
        )
        kernel_fn = reference_mhsa_kernel
    else:
        raise ValueError(f'Unknown multi-head attention kernel: {kernel}')
    o = kernel_fn(q, k, v, mask)
    return o


def reference_mhsa_kernel(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    interpret: bool = False,
) -> jax.Array:
    r"""Reference jax implementation of the multi-head attention distance kernel."""
    del interpret  # Only used with the pallas kernel
    # [batch_len, seq_len, num_heads, seq_len]
    square_mask = mask[:, None, None, :] * mask[:, :, None, None]
    s = jnp.einsum('Biha,Bjha->Bihj', q, k)
    s = jnp.where(square_mask, s, -big_number(s.dtype))
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('Bihj,Bjha->Biha', p, v)
    return o


def mhsa_kernel(
    q_ref,  # Inputs
    k_ref,
    v_ref,
    mask_ref,
    o_ref,  # Outputs
    q_block_len: int | None,
):
    r"""The pallas implementation of the multi-head attention kernel.

    Here pallas grid has already removed the batch and head dimensions.

    Args:
        q_ref: Queries, shape ``(sequence_length, head_dim)``
        k_ref: Keys, shape ``(sequence_length, head_dim)``
        v_ref: Values, shape ``(sequence_length, head_dim)``
        mask_ref: Mask of the q, k, v values, shape ``(sequence_length,)``
        o_ref: Output, shape ``(sequence_length, head_dim)``
        q_block_len: pallas block length
    """
    # q_idx indicates the start index of the current q-block
    q_idx = 0 if q_block_len is None else pl.program_id(1)
    q_block_len = q_block_len or q_ref.shape[0]
    kv_mask = mask_ref[:]
    # q_slice extracts the relevant slice of the q matrix
    q_slice = pl.dslice(q_idx * q_block_len, q_block_len)
    q_mask = pl.load(mask_ref, (q_slice,))
    square_mask = q_mask[:, None] * kv_mask[None, :]
    # Forward pass
    q = q_ref[:, :]
    k = k_ref[:, :]
    v = v_ref[:, :]
    s = jnp.where(square_mask, pl.dot(q, k, trans_b=True), -big_number(q.dtype))
    p = jax.nn.softmax(s, axis=1)
    o = pl.dot(p, v)
    o_ref[:, :] = o
