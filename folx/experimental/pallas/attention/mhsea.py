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
    get_lse_block_spec,
    get_mask_block_spec,
    get_value_or_laplacian_block_spec,
)


def mhsea(
    q: jax.Array,
    k: jax.Array,
    e: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    input_mask: jax.Array,
    kernel: Literal['pallas', 'reference'] = 'pallas',
    interpret: bool = False,
    q_block_len: int | None = None,
    num_warps: int = 2,
    num_stages: int = 2,
) -> jax.Array:
    r"""Pallas implementation of masked multi-head edge attention."""
    del input_mask  # Only used in the forward Laplacian
    batch_len, seq_len, num_heads, head_len = q.shape
    v_dim = v.shape[-1]
    q_block_len, kv_block_len = compute_q_and_kv_block_len(seq_len, q_block_len)

    if kernel == 'pallas':
        kernel_fn = pl.pallas_call(
            partial(mhsea_kernel, q_block_len=q_block_len),
            grid=create_grid(batch_len, seq_len, num_heads, q_block_len),
            in_specs=[
                get_value_or_laplacian_block_spec(seq_len, head_len, q_block_len, True),
                get_value_or_laplacian_block_spec(seq_len, head_len, kv_block_len),
                get_value_or_laplacian_block_spec(seq_len, seq_len, q_block_len, True),
                get_value_or_laplacian_block_spec(seq_len, v_dim, kv_block_len),
                get_mask_block_spec(seq_len, q_block_len),
            ],
            out_specs=[
                get_value_or_laplacian_block_spec(seq_len, v_dim, q_block_len, True),
                get_lse_block_spec(seq_len, q_block_len),
            ],
            out_shape=[
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, v_dim), dtype=v.dtype
                ),  # o
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads), dtype=q.dtype
                ),  # lse
            ],
            compiler_params=dict(
                triton=dict(num_warps=num_warps, num_stages=num_stages)
            ),
            debug=False,
            interpret=interpret,
            name='mhsea',
        )
        o, lse = kernel_fn(q, k, e, v, mask)
        return o  # `lse` is computed but not returned in `mhea`; it's used internally.
    elif kernel == 'reference':
        logging.warning(
            'Passing kernel="reference" to function mhsea is not recommended in production, '
            'as it is very slow. Use kernel="pallas" instead.'
        )
        o = reference_mhsea_kernel(q, k, v, mask, e)
        return o
    else:
        raise ValueError(f'Unknown multi-head attention distance kernel: {kernel}')


def reference_mhsea_kernel(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    edges: jax.Array,
    interpret: bool = False,
) -> jax.Array:
    r"""Reference jax implementation of the multi-head attention distance kernel."""
    del interpret  # Only used with the pallas kernel
    # [batch_len, seq_len, num_heads, seq_len]
    square_mask = mask[:, None, None, :] * mask[:, :, None, None]
    s = jnp.einsum('Biha,Bjha->Bihj', q, k)
    s += edges
    s = jnp.where(square_mask, s, -big_number(s.dtype))
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('Bihj,Bjha->Biha', p, v)
    return o


def mhsea_kernel(
    q_ref,  # Inputs
    k_ref,
    e_ref,
    v_ref,
    mask_ref,
    o_ref,  # Outputs
    lse_ref,
    q_block_len: int | None,
):
    r"""The pallas implementation of the multi-head edge attention kernel.

    Here pallas grid has already removed the batch and head dimensions.

    Args:
        q_ref: Queries, shape ``(sequence_length, head_dim)``
        k_ref: Keys, shape ``(sequence_length, head_dim)``
        e_ref: Edges, shape ``(sequence_length, sequence_length)``
        v_ref: Values, shape ``(sequence_length, head_dim)``
        mask_ref: Mask of the q, k, v values, shape ``(sequence_length,)``
        o_ref: Output, shape ``(sequence_length, head_dim)``
        lse_ref: Output, shape ``(sequence_length, head_dim)``
        q_block_len: pallas block length
        epsilon: for distances
    """
    q_idx = 0 if q_block_len is None else pl.program_id(1)
    q_block_len = q_block_len or q_ref.shape[0]
    kv_mask = mask_ref[:]
    q_slice = pl.Slice(q_idx * q_block_len, q_block_len)
    q_mask = pl.load(mask_ref, (q_slice,))
    square_mask = q_mask[:, None] * kv_mask[None, :]
    # Forward pass
    q = jnp.where(q_mask[:, None], q_ref[:, :], 0.0)
    k = jnp.where(kv_mask[:, None], k_ref[:, :], 0.0)
    e = e_ref[:, :]
    v = jnp.where(kv_mask[:, None], v_ref[:, :], 0.0)
    s = jnp.where(square_mask, pl.dot(q, k, trans_b=True) + e, -big_number(q.dtype))
    max_val = jnp.max(
        s, axis=1, keepdims=False
    )  # Take the max along the axis to stabilize
    lse = max_val + jnp.log(jnp.sum(jnp.exp(s - max_val[:, None]), axis=1))
    p = jnp.exp(s - lse[:, None])
    lse_ref[:] = lse
    o = pl.dot(p, v)
    o_ref[:, :] = o
