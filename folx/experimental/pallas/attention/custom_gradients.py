import logging
from functools import partial
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .mhsa import mhsa_kernel, reference_mhsa_kernel
from .mhsea import mhsea_kernel, reference_mhsea_kernel
from .utils import (
    big_number,
    compute_q_and_kv_block_len,
    create_grid,
    get_lse_block_spec,
    get_mask_block_spec,
    get_value_or_laplacian_block_spec,
    sum_columns,
)

#######################################################################################################
# Multi-head attention VJP
#######################################################################################################


def mhsa_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    input_mask: jax.Array,
    kernel: Literal['pallas', 'reference'],
    interpret: bool,
    q_block_len: int | None,
    num_warps: int,
    num_stages: int,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
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
            name='mhsa_forward',
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
    return o, (q, k, v, mask)


def mhsa_backward(
    kernel: Literal['pallas', 'reference'],
    interpret: bool,
    q_block_len: int | None,
    num_warps: int,
    num_stages: int,
    fwd_cache: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    o_vjp: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, None, None]:
    assert q_block_len is None, 'Q blocking is not implemented in backward'
    q, k, v, mask = fwd_cache
    batch_len, seq_len, num_heads, head_len = q.shape
    q_block_len, kv_block_len = compute_q_and_kv_block_len(seq_len, q_block_len)

    if kernel == 'pallas':
        kernel_fn = pl.pallas_call(
            mhsa_backward_kernel,
            grid=create_grid(batch_len, seq_len, num_heads, q_block_len),
            in_specs=[
                get_value_or_laplacian_block_spec(seq_len, head_len, q_block_len),
                get_value_or_laplacian_block_spec(seq_len, head_len, kv_block_len),
                get_value_or_laplacian_block_spec(seq_len, head_len, kv_block_len),
                get_mask_block_spec(seq_len, q_block_len),
                get_value_or_laplacian_block_spec(seq_len, head_len, q_block_len),
            ],
            out_specs=[
                get_value_or_laplacian_block_spec(seq_len, head_len, q_block_len),
                get_value_or_laplacian_block_spec(seq_len, head_len, kv_block_len),
                get_value_or_laplacian_block_spec(seq_len, head_len, kv_block_len),
            ],
            out_shape=[
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, head_len), dtype=q.dtype
                ),
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, head_len), dtype=q.dtype
                ),
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, head_len), dtype=q.dtype
                ),
            ],
            compiler_params=dict(
                triton=dict(num_warps=num_warps, num_stages=num_stages)
            ),
            debug=False,
            interpret=interpret,
            name='mhsa_backward',
        )
    elif kernel == 'reference':
        kernel_fn = reference_mhsa_backward_kernel
    else:
        raise ValueError(f'Unknown multi-head attention kernel: {kernel}')
    dq, dk, dv = kernel_fn(q, k, v, mask, o_vjp)
    return dq, dk, dv, None, None


def reference_mhsa_backward_kernel(
    q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array, o_vjp: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""Reference jax implementation of the multi-head attention backward kernel."""
    # [batch_size, seq_len, num_heads, seq_len]
    q = jnp.where(mask[:, :, None, None], q, 0.0)
    square_mask = mask[:, None, None, :] * mask[:, :, None, None]
    s = jnp.einsum('Biha,Bjha->Bihj', q, k)
    s = jnp.where(square_mask, s, -big_number(q.dtype))
    p = jax.nn.softmax(s, axis=-1)

    # Compute the VJPs
    p_vjp = jnp.einsum('Biha,Bjha->Bihj', o_vjp, v)
    q_vjp = jnp.einsum('Bkha,Bihk,Bihk->Biha', k, p, p_vjp) - jnp.einsum(
        'Bmha,Bihk,Bihm,Bihk->Biha', k, p, p, p_vjp
    )
    k_vjp = jnp.einsum('Bjha,Bjhi,Bjhi->Biha', q, p, p_vjp) - jnp.einsum(
        'Bjha,Bjhk,Bjhi,Bjhk->Biha', q, p, p, p_vjp
    )
    v_vjp = jnp.einsum('Bjhi,Bjha->Biha', p, o_vjp)

    return q_vjp, k_vjp, v_vjp


def mhsa_backward_kernel(
    q_ref,  # Inputs
    k_ref,
    v_ref,
    mask_ref,
    o_vjp_ref,
    q_vjp_ref,  # Outputs
    k_vjp_ref,
    v_vjp_ref,
):
    r"""The pallas implementation of the backward of the multi-head attention kernel.

    Here pallas grid has already removed the batch and head dimensions.

    Args:
        q_ref: Queries, shape ``(sequence_length, head_dim)``
        k_ref: Keys, shape ``(sequence_length, head_dim)``
        v_ref: Values, shape ``(sequence_length, head_dim)``
        mask_ref: Mask of the q, k, v values, shape ``(sequence_length,)``
        o_vjp_ref: VJP of the output of MHA, shape ``(sequence_length, head_dim)``
        q_vjp_ref: output, VJP of the queries, shape ``(sequence_length, head_dim)``
        k_vjp_ref: output, VJP of the keys, shape ``(sequence_length, head_dim)``
        v_vjp_ref: output, VJP of the values, shape ``(sequence_length, head_dim)``
    """
    mask = mask_ref[:]
    square_mask = mask[:, None] * mask[None, :]
    # Recompute the output to save memory
    q = jnp.where(mask[:, None], q_ref[:, :], 0.0)
    k = jnp.where(mask[:, None], k_ref[:, :], 0.0)
    v = jnp.where(mask[:, None], v_ref[:, :], 0.0)
    s = jnp.where(square_mask, pl.dot(q, k, trans_b=True), -big_number(q.dtype))
    p = jax.nn.softmax(s)

    # Compute the VJPs
    o_vjp = o_vjp_ref[:, :]

    # v_vjp
    v_vjp = pl.dot(p, o_vjp, trans_a=True)
    v_vjp_ref[:, :] = v_vjp

    # q_vjp
    lo_v_p = pl.dot(o_vjp, v, trans_b=True) * p
    ## First term
    q_vjp = pl.dot(lo_v_p, k)
    ## Second term
    pk = pl.dot(p, k)
    q_vjp -= pk * sum_columns(lo_v_p)
    q_vjp_ref[:, :] = q_vjp

    # k_vjp
    ## First term
    k_vjp = pl.dot(lo_v_p.T, q)
    ## Second term
    p_vjp = pl.dot(o_vjp, v, trans_b=True)
    k_vjp -= pl.dot((p * sum_columns(p_vjp * p)), q, trans_a=True)
    k_vjp_ref[:, :] = k_vjp


#######################################################################################################
# Multi-head self edge attention VJP
#######################################################################################################


def mhsea_forward(
    q: jax.Array,
    k: jax.Array,
    e: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    input_mask: jax.Array,
    kernel: Literal['pallas', 'reference'] = 'pallas',
    interpret: bool = False,
    q_block_len: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tuple[
    jax.Array,
    Tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array | None,
        jax.Array | None,
    ],
]:
    r"""Pallas implementation of masked multi-head self edge attention, forward pass."""
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
                ),
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads), dtype=v.dtype
                ),
            ],
            compiler_params=dict(
                triton=dict(num_warps=num_warps, num_stages=num_stages)
            ),
            debug=False,
            interpret=interpret,
            name='mhea_forward',
        )
        o, lse = kernel_fn(q, k, e, v, mask)
        return o, (q, k, e, v, mask, lse, o)
    elif kernel == 'reference':
        logging.warning(
            'Passing kernel="reference" to function mhsea is not recommended in production, '
            'as it is very slow. Use kernel="pallas" instead.'
        )
        return reference_mhsea_kernel(q, k, v, mask, edges=e), (
            q,
            k,
            e,
            v,
            mask,
            None,
            None,
        )
    else:
        raise ValueError(f'Unknown multi-head attention distance kernel: {kernel}')


def mhsea_backward(
    kernel: Literal['pallas', 'reference'],
    interpret: bool,
    q_block_len: int | None,
    num_warps: int,
    num_stages: int,
    fwd_cache: Tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array | None,
        jax.Array | None,
    ],
    o_vjp: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, None, None]:
    r"""Pallas implementation of masked multi-head self edge attention, backward pass."""
    q, k, e, v, mask, lse, o = fwd_cache
    batch_len, seq_len, num_heads, head_len = q.shape
    block_len = seq_len if q_block_len is None else q_block_len

    if kernel == 'pallas':
        dq, de = pl.pallas_call(
            partial(mhsea_q_vjp_kernel, block_len=block_len),
            grid=(batch_len, seq_len // block_len, num_heads),
            in_specs=[
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, j, k, 0),
                    block_shape=(None, block_len, None, head_len),
                ),  # q
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, 0, k, 0),
                    block_shape=(None, seq_len, None, head_len),
                ),  # k
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, j, k, 0),
                    block_shape=(None, block_len, None, seq_len),
                ),  # e
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, 0, k, 0),
                    block_shape=(None, seq_len, None, head_len),
                ),  # v
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, 0), block_shape=(None, seq_len)
                ),  # mask
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, j, k),
                    block_shape=(None, block_len, None),
                ),  # lse
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, j, k, 0),
                    block_shape=(None, block_len, None, head_len),
                ),  # o
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, j, k, 0),
                    block_shape=(None, block_len, None, head_len),
                ),  # o_vjp
            ],
            out_specs=[
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, j, k, 0),
                    block_shape=(None, block_len, None, head_len),
                ),  # dq
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, j, k, 0),
                    block_shape=(None, block_len, None, seq_len),
                ),  # de
            ],
            out_shape=[
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, head_len), dtype=q.dtype
                ),
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, seq_len), dtype=e.dtype
                ),
            ],
            compiler_params=dict(
                triton=dict(num_warps=num_warps, num_stages=num_stages)
            ),
            debug=False,
            interpret=interpret,
            name='mhsea_backward_q_vjp',
        )(q, k, e, v, mask, lse, o, o_vjp)
        dk, dv = pl.pallas_call(
            partial(mhsea_kv_vjp_kernel, block_len=block_len),
            grid=(batch_len, seq_len // block_len, num_heads),
            in_specs=[
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, 0, k, 0),
                    block_shape=(None, seq_len, None, head_len),
                ),  # q
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, j, k, 0),
                    block_shape=(None, block_len, None, head_len),
                ),  # k
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, 0, k, j),
                    block_shape=(None, seq_len, None, block_len),
                ),  # e
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, j, k, 0),
                    block_shape=(None, block_len, None, head_len),
                ),  # v
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, 0), block_shape=(None, seq_len)
                ),  # mask
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, 0, k),
                    block_shape=(None, seq_len, None),
                ),  # lse
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, 0, k, 0),
                    block_shape=(None, seq_len, None, head_len),
                ),  # o
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, 0, k, 0),
                    block_shape=(None, seq_len, None, head_len),
                ),  # o_vjp
            ],
            out_specs=[
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, j, k, 0),
                    block_shape=(None, block_len, None, head_len),
                ),  # dk
                pl.BlockSpec(
                    index_map=lambda i, j, k: (i, j, k, 0),
                    block_shape=(None, block_len, None, head_len),
                ),  # dv
            ],
            out_shape=[
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, head_len), dtype=k.dtype
                ),
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, head_len), dtype=v.dtype
                ),
            ],
            compiler_params=dict(
                triton=dict(num_warps=num_warps, num_stages=num_stages)
            ),
            debug=False,
            interpret=interpret,
            name='mhsea_backward_kv_vjp',
        )(q, k, e, v, mask, lse, o, o_vjp)
    elif kernel == 'reference':
        dq, dk, de, dv = reference_mhsea_backward_kernel(q, k, e, v, mask, o_vjp)
    else:
        raise ValueError(f'Unknown multi-head attention kernel: {kernel}')
    return dq, dk, de, dv, None, None


def mhsea_q_vjp_kernel(
    q_ref,  # Inputs
    k_ref,
    e_ref,
    v_ref,
    mask_ref,
    lse_ref,
    o_ref,
    o_vjp_ref,
    q_vjp_ref,  # Outputs
    e_vjp_ref,
    block_len,
):
    """Pallas implementation of the backward pass of the multi-head self edge attention
    kernel to compute `q_vjp` `e_vjp`.

    Here Pallas grid has already removed the batch and head dimensions.

    Args:
        q_ref: Queries, shape `(block_length, head_dim)`
        k_ref: Keys, shape `(sequence_length, head_dim)`
        e_ref: Edge information, shape `(sequence_length, sequence_length)`
        v_ref: Values, shape `(sequence_length, head_dim)`
        mask_ref: Mask, shape `(sequence_length,)`
        lse_ref: Cached logsumexp values, shape `(sequence_length,)`
        o_ref: Cached outputs, shape `(sequence_length, head_dim)`
        o_vjp_ref: VJP of the output of MHEA, shape `(sequence_length, head_dim)`
        q_vjp_ref: Output, VJP of the queries, shape `(block_length, head_dim)`
        e_vjp_ref: Output, VJP of the edges, shape `(sequence_length, sequence_length)`
        block_len: Block length for parallel computation
    """
    qix = pl.program_id(1)
    q_vjp = jnp.zeros(q_vjp_ref.shape, dtype=q_vjp_ref.dtype)
    q_slice = pl.Slice(qix * block_len, block_len)

    def _kaxis_loop(kix, q_vjp):
        mask_q = mask_ref[q_slice]
        k_slice = pl.Slice(kix * block_len, block_len)
        mask_k = mask_ref[k_slice]
        square_mask = mask_q[:, None] * mask_k[None, :]
        q = jnp.where(mask_q[:, None], q_ref[:, :], 0.0)
        k = jnp.where(mask_k[:, None], k_ref[k_slice, :], 0.0)
        v = jnp.where(mask_k[:, None], v_ref[k_slice, :], 0.0)
        e = e_ref[:, k_slice]
        lse = lse_ref[:]

        s = jnp.where(square_mask, pl.dot(q, k, trans_b=True) + e, -big_number(q.dtype))

        p = jnp.exp(s - lse[:, None])
        o = o_ref[:, :]  # caching, because o requires sum over k

        # Compute the VJP
        o_vjp = jnp.where(mask_q[:, None], o_vjp_ref[:, :], 0.0)

        s_vjp = (pl.dot(o_vjp, v, trans_b=True) - sum_columns(o * o_vjp)) * p
        s_vjp *= square_mask

        # Sum over k-axis
        q_vjp_kblock = pl.dot(s_vjp, k)
        q_vjp += q_vjp_kblock
        e_vjp_ref[:, k_slice] = s_vjp

        return q_vjp

    q_vjp = jax.lax.fori_loop(0, k_ref.shape[0] // block_len, _kaxis_loop, q_vjp)
    q_vjp_ref[:, :] = q_vjp


def mhsea_kv_vjp_kernel(
    q_ref,  # Inputs
    k_ref,
    e_ref,
    v_ref,
    mask_ref,
    lse_ref,
    o_ref,
    o_vjp_ref,
    k_vjp_ref,  # Outputs
    v_vjp_ref,
    block_len,
):
    r"""The Pallas implementation of the backward pass of the multi-head edge attention kernel.

    Computes `k_vjp`, `v_vjp` using block-based computation.

    Args:
        q_ref: Queries, shape ``(sequence_length, head_dim)``
        k_ref: Keys, shape ``(sequence_length, head_dim)``
        e_ref: Edges, shape ``(sequence_length, sequence_length)``
        v_ref: Values, shape ``(sequence_length, head_dim)``
        mask_ref: Mask of the q, k, v values, shape ``(sequence_length,)``
        lse_ref: Cached logsumexp values, shape ``(sequence_length,)``
        o_ref: Cached outputs, shape ``(sequence_length, head_dim)``
        o_vjp_ref: VJP of the output of MHA, shape ``(sequence_length, head_dim)``
        k_vjp_ref: Output, VJP of the keys, shape ``(sequence_length, head_dim)``
        e_vjp_ref: Output, VJP of the edges, shape ``(sequence_length, sequence_length)``
        v_vjp_ref: Output, VJP of the values, shape ``(sequence_length, head_dim)``
        block_len: Block length for tiled computation
    """
    kix = pl.program_id(1)
    k_vjp = jnp.zeros((block_len, k_vjp_ref.shape[-1]), dtype=k_vjp_ref.dtype)
    v_vjp = jnp.zeros((block_len, v_vjp_ref.shape[-1]), dtype=v_vjp_ref.dtype)
    k_slice = pl.Slice(kix * block_len, block_len)

    def _qaxis_loop(qix, store):
        k_vjp, v_vjp = store
        q_slice = pl.Slice(qix * block_len, block_len)
        mask_q = mask_ref[q_slice]
        mask_k = mask_ref[k_slice]
        square_mask = mask_q[:, None] * mask_k[None, :]
        q = jnp.where(mask_q[:, None], q_ref[q_slice, :], 0.0)
        k = jnp.where(mask_k[:, None], k_ref[:, :], 0.0)
        v = jnp.where(mask_k[:, None], v_ref[:, :], 0.0)
        e = e_ref[q_slice, :]
        lse = lse_ref[q_slice]

        s = jnp.where(square_mask, pl.dot(q, k, trans_b=True) + e, -big_number(q.dtype))

        p = jnp.exp(s - lse[:, None])
        o = o_ref[q_slice, :]  # caching, because o requires sum over k

        # Compute the VJPs
        o_vjp = jnp.where(mask_q[:, None], o_vjp_ref[q_slice, :], 0.0)

        s_vjp = (pl.dot(o_vjp, v, trans_b=True) - sum_columns(o * o_vjp)) * p
        s_vjp *= square_mask

        # These two are summed over the q-axis
        k_vjp += pl.dot(s_vjp, q, trans_a=True)
        v_vjp += pl.dot(p, o_vjp, trans_a=True)

        return k_vjp, v_vjp

    k_vjp, v_vjp = jax.lax.fori_loop(
        0, q_ref.shape[0] // block_len, _qaxis_loop, (k_vjp, v_vjp)
    )

    k_vjp_ref[:, :] = k_vjp
    v_vjp_ref[:, :] = v_vjp


def reference_mhsea_backward_kernel(
    q: jax.Array,
    k: jax.Array,
    e: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    o_vjp: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Reference jax implementation of the multi-head self edge attention backward kernel."""
    square_mask = mask[:, None, None, :] * mask[:, :, None, None]
    # [batch_len, seq_len, num_heads, head_len]
    s = jnp.einsum('Biha,Bjha->Bihj', q, k) + e
    # Bijh
    s = jnp.where(square_mask, s, -big_number(s.dtype))
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('Bihj,Bjha->Biha', p, v)

    # Compute the VJPs
    s_vjp = jnp.einsum('Bihj,Bjha,Biha->Bihj', p, v, o_vjp) - jnp.einsum(
        'Bihj,Biha,Biha->Bihj', p, o, o_vjp
    )
    s_vjp *= square_mask
    q_vjp = jnp.einsum('Bihj,Bjha->Biha', s_vjp, k)
    k_vjp = jnp.einsum('Bihj,Biha->Bjha', s_vjp, q)
    v_vjp = jnp.einsum('Bjhi,Bjha->Biha', p, o_vjp)

    return q_vjp, k_vjp, s_vjp, v_vjp
