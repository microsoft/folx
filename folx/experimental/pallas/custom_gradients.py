from functools import partial
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .mha import mha_kernel, reference_mha_kernel
from .utils import (
    compute_q_and_kv_block_len,
    create_grid,
    get_mask_block_spec,
    get_value_or_laplacian_block_spec,
    sum_columns,
)

#######################################################################################################
# Multi-head attention VJP
#######################################################################################################


def mha_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    input_mask: jax.Array,
    kernel: Literal["pallas", "reference"] = "pallas",
    interpret: bool = False,
    q_block_len: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    del input_mask  # Only used in the forward Laplacian
    batch_len, seq_len, num_heads, head_len = q.shape
    q_block_len, kv_block_len = compute_q_and_kv_block_len(seq_len, q_block_len)

    if kernel == "pallas":
        kernel_fn = pl.pallas_call(
            partial(mha_kernel, q_block_len=q_block_len),
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
            compiler_params=dict(triton=dict(num_warps=num_warps, num_stages=num_stages)),
            debug=False,
            interpret=interpret,
            name="mha_forward",
        )
    elif kernel == "reference":
        kernel_fn = reference_mha_kernel
    else:
        raise ValueError(f"Unknown multi-head attention kernel: {kernel}")
    o = kernel_fn(q, k, v, mask)
    return o, (q, k, v, mask)


def mha_backward(
    kernel: Literal["pallas", "reference"],
    interpret: bool,
    q_block_len: int | None,
    num_warps: int,
    num_stages: int,
    fwd_cache: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    o_vjp: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, None, None]:
    assert q_block_len is None, "Q blocking is not implemented in backward"
    q, k, v, mask = fwd_cache
    batch_len, seq_len, num_heads, head_len = q.shape
    q_block_len, kv_block_len = compute_q_and_kv_block_len(seq_len, q_block_len)

    if kernel == "pallas":
        kernel_fn = pl.pallas_call(
            mha_backward_kernel,
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
            compiler_params=dict(triton=dict(num_warps=num_warps, num_stages=num_stages)),
            debug=False,
            interpret=interpret,
            name="mha_backward",
        )
    elif kernel == "reference":
        kernel_fn = reference_mha_backward_kernel
    else:
        raise ValueError(f"Unknown multi-head attention kernel: {kernel}")
    dq, dk, dv = kernel_fn(q, k, v, mask, o_vjp)
    return dq, dk, dv, None, None


def reference_mha_backward_kernel(
    q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array, o_vjp: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""Reference jax implementation of the multi-head attention backward kernel."""
    # [batch_size, seq_len, num_heads, seq_len]
    q = jnp.where(mask[:, :, None, None], q, 0.0)
    square_mask = mask[:, None, None, :] * mask[:, :, None, None]
    s = jnp.einsum("Biha,Bjha->Bihj", q, k)
    s = jnp.where(square_mask, s, -1e20)
    p = jax.nn.softmax(s, axis=-1)

    # Compute the VJPs
    p_vjp = jnp.einsum("Biha,Bjha->Bihj", o_vjp, v)
    q_vjp = jnp.einsum("Bkha,Bihk,Bihk->Biha", k, p, p_vjp) - jnp.einsum(
        "Bmha,Bihk,Bihm,Bihk->Biha", k, p, p, p_vjp
    )
    k_vjp = jnp.einsum("Bjha,Bjhi,Bjhi->Biha", q, p, p_vjp) - jnp.einsum(
        "Bjha,Bjhk,Bjhi,Bjhk->Biha", q, p, p, p_vjp
    )
    v_vjp = jnp.einsum("Bjhi,Bjha->Biha", p, o_vjp)

    return q_vjp, k_vjp, v_vjp


def mha_backward_kernel(
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
    s = jnp.where(square_mask, pl.dot(q, k, trans_b=True), -1e20)
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
