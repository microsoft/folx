import logging
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from folx import forward_laplacian
from folx.api import FwdJacobian, FwdLaplArray

from .mhsa import reference_mhsa_kernel
from .mhsea import reference_mhsea_kernel
from .utils import (
    big_number,
    compute_q_and_kv_block_len,
    create_grid,
    get_input_mask_block_spec,
    get_jacobian_block_spec,
    get_mask_block_spec,
    get_value_or_laplacian_block_spec,
    sum_columns,
)


def mhsa_forward_laplacian(
    args: Tuple[FwdLaplArray, FwdLaplArray, FwdLaplArray, jax.Array, jax.Array],
    kwargs: Dict[str, Any],
    sparsity_threshold: int,
) -> FwdLaplArray:
    r"""Forward laplacian of attention, to be used with folx.

    This function should be passed to ``folx.register_function``, as the custom forward
    Laplacian of multi-head attention.

    The default settings, for ``q_block_len``, ``num_warps``, and ``num_stages`` reflect
    the results of internal benchmarking.

    The input ``FwdLaplArray``s should have the following shapes:
        - x: ``(batch_size, sequence_length, num_heads, head_dim)``
        - jacobian: ``(input_dim, batch_size, sequence_length, num_heads, head_dim)``,
            where ``input_dim`` is the original input dimension of the model with respect to
            which the forward Laplacian is computed (usually ``3 * n_elec``).
        - laplacian: ``(batch_size, sequence_length, num_heads, head_dim)``

    Args:
        args: tuple of ``q``, ``k``, ``v``, ``mask``, and ``input_mask``.
            - q: ``FwdLaplArray`` of queries
            - k: ``FwdLaplArray`` of keys
            - v: ``FwdLaplArray`` of values
            - mask: mask of the q, k, v values, shape ``(batch_size, sequence_length)``.
            - input_mask: mask of the original input to the model, with respect to which
            -     the forward Laplacian is computed, shape ``(input_dim, batch_size)``.
        kwargs:
            - kernel (str): Default is ``pallas``.
                - folx: the vanilla folx kernel is used.
                - reference: the reference jax kernel is used.
                - pallas: the pallas kernel is used.
            - interpret: If ``True``, the pallas kernel is executed in interpret mode,
                which allows it to be executed e.g. on a CPU (slow).
            - q_block_len (int | None): If ``None``, there is no blocking of the query
                array, otherwise it's blocked into blocks of length ``q_block_len``.
                Default is 16.
            - num_warps (int): The number of threads to execute a single instance of the
                kernel with. Default is `sequence_length // 8`.
            - num_stages (int): The number of stages??. Default is 1.
        sparsity_threshold: Sparsity threshold of folx, currently ignored.
    """
    del sparsity_threshold
    q, k, v, mask, input_mask = args
    assert len(input_mask) == len(q.jacobian.dense_array)
    input_len, batch_len, seq_len, num_heads, head_len = q.jacobian.dense_array.shape

    kernel = kwargs.get('kernel', 'pallas')
    interpret = kwargs.get('interpret', False)
    q_block_len = kwargs.get('q_block_len', 16)
    num_warps = kwargs.get('num_warps', seq_len // 8)
    num_stages = kwargs.get('num_stages', 1)

    q_block_len, kv_block_len = compute_q_and_kv_block_len(seq_len, q_block_len)

    if kernel == 'folx':
        kernel_fn = folx_mhsa_forward_laplacian_kernel
    elif kernel == 'reference':
        logging.warning(
            'Passing kernel="reference" to function mhsa is not recommended in production, '
            'as it is very slow. Use kernel="pallas" instead.'
        )
        kernel_fn = reference_mhsa_forward_laplacian_kernel
    elif kernel == 'pallas':
        kernel_fn = pl.pallas_call(
            partial(mhsa_forward_laplacian_kernel, q_block_len=q_block_len),
            grid=create_grid(batch_len, seq_len, num_heads, q_block_len),
            in_specs=[
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, q_block_len, True
                ),  # q.x
                get_jacobian_block_spec(
                    input_len, seq_len, head_len, q_block_len, True
                ),  # q.jacobian
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, q_block_len, True
                ),  # q.laplacian
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, kv_block_len
                ),  # k.x
                get_jacobian_block_spec(
                    input_len, seq_len, head_len, kv_block_len
                ),  # k.jacobian
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, kv_block_len
                ),  # k.laplacian
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, kv_block_len
                ),  # v.x
                get_jacobian_block_spec(
                    input_len, seq_len, head_len, kv_block_len
                ),  # v.jacobian
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, kv_block_len
                ),  # v.laplacian
                get_mask_block_spec(seq_len, q_block_len),  # mask
                get_input_mask_block_spec(input_len, q_block_len),  # input_mask
            ],
            out_specs=[
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, q_block_len, True
                ),  # o.x
                get_jacobian_block_spec(
                    input_len, seq_len, head_len, q_block_len, True
                ),  # o.jacobian
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, q_block_len, True
                ),  # o.laplacian
            ],
            out_shape=[
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, head_len),
                    dtype=q.dtype,  # o.x
                ),
                jax.ShapeDtypeStruct(
                    shape=(
                        input_len,
                        batch_len,
                        seq_len,
                        num_heads,
                        head_len,
                    ),  # o.jacobian
                    dtype=q.dtype,
                ),
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, head_len),
                    dtype=q.dtype,  # o.laplacian
                ),
            ],
            compiler_params=dict(
                triton=dict(num_warps=num_warps, num_stages=num_stages)
            ),
            debug=False,
            interpret=interpret,
            name='mhsa_forward_laplacian',
        )
    else:
        raise ValueError(f'Unknown forward Laplacian attention kernel: {kernel}')
    # TODO: Can we avoid calling `.dense_array` on the Jacobians and instead use sparse matrices here?
    # This would help with LapNet-like attention
    x, jacobian, laplacian = kernel_fn(
        q.x,
        q.jacobian.dense_array,
        q.laplacian,
        k.x,
        k.jacobian.dense_array,
        k.laplacian,
        v.x,
        v.jacobian.dense_array,
        v.laplacian,
        mask,
        input_mask,
    )
    return FwdLaplArray(x, FwdJacobian(jacobian, None), laplacian)


@jax.jit
def folx_mhsa_forward_laplacian_kernel(
    q: jax.Array,
    q_jac: jax.Array,
    q_lap: jax.Array,
    k: jax.Array,
    k_jac: jax.Array,
    k_lap: jax.Array,
    v: jax.Array,
    v_jac: jax.Array,
    v_lap: jax.Array,
    mask: jax.Array,
    input_mask: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""Vanilla folx implementation of the multi-head attention forward Laplacian kernel."""
    q_fwd_lap = FwdLaplArray(
        q, FwdJacobian.from_dense(q_jac * input_mask[..., None, None, None]), q_lap
    )
    k_fwd_lap = FwdLaplArray(
        k, FwdJacobian.from_dense(k_jac * input_mask[..., None, None, None]), k_lap
    )
    v_fwd_lap = FwdLaplArray(
        v, FwdJacobian.from_dense(v_jac * input_mask[..., None, None, None]), v_lap
    )
    o_fwd_lap = forward_laplacian(reference_mhsa_kernel)(
        q_fwd_lap, k_fwd_lap, v_fwd_lap, mask
    )  # type: ignore
    return o_fwd_lap.x, o_fwd_lap.jacobian.dense_array, o_fwd_lap.laplacian


@jax.jit
def reference_mhsa_forward_laplacian_kernel(
    q: jax.Array,
    q_jac: jax.Array,
    q_lap: jax.Array,
    k: jax.Array,
    k_jac: jax.Array,
    k_lap: jax.Array,
    v: jax.Array,
    v_jac: jax.Array,
    v_lap: jax.Array,
    mask: jax.Array,
    input_mask: jax.Array,
):
    r"""Reference jax implementation of the multi-head attention forward Laplacian kernel."""
    # [batch_size, seq_len, num_heads, seq_len]
    square_mask = mask[:, None, None, :] * mask[:, :, None, None]
    # [input_dim, batch_size, seq_len, num_heads, head_dim]
    coordinate_and_electron_mask = (
        input_mask[:, :, None, None, None] * mask[None, :, :, None, None]
    )
    # [batch_size, seq_len, num_heads, head_dim]
    qkv_mask = mask[:, :, None, None]
    q_jac = jnp.where(coordinate_and_electron_mask, q_jac, 0.0)
    k_jac = jnp.where(coordinate_and_electron_mask, k_jac, 0.0)
    v_jac = jnp.where(coordinate_and_electron_mask, v_jac, 0.0)
    q_lap = jnp.where(qkv_mask, q_lap, 0.0)
    k_lap = jnp.where(qkv_mask, k_lap, 0.0)
    v_lap = jnp.where(qkv_mask, v_lap, 0.0)
    # Forward
    s = jnp.einsum('Bnhd,BNhd->BnhN', q, k)
    s = jnp.where(square_mask, s, -big_number(s.dtype))
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('BnhN,BNhd->Bnhd', p, v)

    # Jacobian
    delta = jnp.eye(q.shape[1])
    delta_minus_p = delta[None, None, :, None, :] - jnp.moveaxis(
        p[:, :, :, :, None], 2, 3
    )
    first_term = jnp.einsum(
        'pBihb,Bmhb,Bihk,Bimhk,Bkha->pBiha', q_jac, k, p, delta_minus_p, v
    )
    second_term = jnp.einsum(
        'pBmhb,Bihb,Bihk,Bimhk,Bkha->pBiha', k_jac, q, p, delta_minus_p, v
    )
    third_term = jnp.einsum('pBjha,Bihj->pBiha', v_jac, p)
    o_jac = first_term + second_term + third_term

    # Laplacian
    ## J(f)*L(g)
    o_q = jnp.einsum('Bihb,Bmhb,Bihk,Bimhk,Bkha->Biha', q_lap, k, p, delta_minus_p, v)
    o_k = jnp.einsum('Bjhb,Bihb,Bihk,Bijhk,Bkha->Biha', k_lap, q, p, delta_minus_p, v)
    o_v = jnp.einsum('Bjha,Bihj->Biha', v_lap, p)

    ## tr(J(g) H(f) J(g)^T)
    ### P^K
    o_vqr2 = jnp.einsum(
        'pBihc,pBkhb,Bmhb,Bkhi,Bkmhi->Bkhc', v_jac, q_jac, k, p, delta_minus_p
    )
    o_vkr2 = jnp.einsum(
        'pBihc,pBjhb,Bkhb,Bkhi,Bkjhi->Bkhc', v_jac, k_jac, q, p, delta_minus_p
    )
    o_qqr2 = jnp.einsum(
        'pBkha,pBkhb,Bmhb,Bnha,Bkhl,Bknhl,Bkmhl,Blhc->Bkhc',
        q_jac,
        q_jac,
        k,
        k,
        p,
        delta_minus_p,
        delta_minus_p,
        v,
    ) - jnp.einsum(
        'pBkha,pBkhb,Bmhb,Bkhl,Bnha,Bkhm,Bknhm,Blhc->Bkhc',
        q_jac,
        q_jac,
        k,
        p,
        k,
        p,
        delta_minus_p,
        v,
    )
    o_qkr2 = (
        jnp.einsum(
            'pBkha,pBjha,Bkhl,Bkjhl,Blhc->Bkhc', q_jac, k_jac, p, delta_minus_p, v
        )
        + jnp.einsum(
            'pBkha,pBjhb,Bkhb,Bmha,Bkhl,Bkmhl,Bkjhl,Blhc->Bkhc',
            q_jac,
            k_jac,
            q,
            k,
            p,
            delta_minus_p,
            delta_minus_p,
            v,
        )
        - jnp.einsum(
            'pBkha,pBjhb,Bkhb,Bkhl,Bmha,Bkhj,Bkmhj,Blhc->Bkhc',
            q_jac,
            k_jac,
            q,
            p,
            k,
            p,
            delta_minus_p,
            v,
        )
    )
    o_kkr2 = jnp.einsum(
        'pBiha,Bkhb,Bkha,Bkhl,Bkihl,Bkjhl,Blhc,pBjhb->Bkhc',
        k_jac,
        q,
        q,
        p,
        delta_minus_p,
        delta_minus_p,
        v,
        k_jac,
    ) - jnp.einsum(
        'pBiha,Bkhb,Bkhl,Bkha,Bkhj,Bkihj,Blhc,pBjhb->Bkhc',
        k_jac,
        q,
        p,
        q,
        p,
        delta_minus_p,
        v,
        k_jac,
    )
    o_vvr2 = 0
    o_lap = o_q + o_k + o_v + o_qqr2 + o_kkr2 + o_vvr2 + 2 * (o_vqr2 + o_qkr2 + o_vkr2)
    return o, o_jac, o_lap


def mhsa_forward_laplacian_kernel(
    q_x_ref,  # Inputs
    q_jac_ref,
    q_lap_ref,
    k_x_ref,
    k_jac_ref,
    k_lap_ref,
    v_x_ref,
    v_jac_ref,
    v_lap_ref,
    mask_ref,
    input_mask_ref,
    o_x_ref,  # Outputs
    o_jac_ref,
    o_lap_ref,
    q_block_len: int | None,
):
    r"""The pallas implementation of the multi-head attention forward Laplacian Kernel.

    Here pallas grid has already removed the batch and head dimensions.

    Args:
        q_x_ref: Queries, shape ``(sequence_length, head_dim)``
        q_jac_ref: Jacobian of queries, shape ``(input_dim, sequence_length, head_dim)``
        q_lap_ref: Laplacian of queries, shape ``(sequence_length, head_dim)``
        k_x_ref: Keys, shape ``(sequence_length, head_dim)``
        k_jac_ref: Jacobian of keys, shape ``(input_dim, sequence_length, head_dim)``
        k_lap_ref: Laplacian of keys, shape ``(sequence_length, head_dim)``
        v_x_ref: Values, shape ``(sequence_length, head_dim)``
        v_jac_ref: Jacobian of values, shape ``(input_dim, sequence_length, head_dim)``
        v_lap_ref: Laplacian of values, shape ``(sequence_length, head_dim)``
        mask_ref: Mask of the q, k, v values, shape ``(sequence_length,)``
        input_mask_ref: Mask of the original inputs to the model, shape ``(input_dim,)``
        o_x_ref: Output, shape ``(sequence_length, head_dim)``
        o_jac_ref: Jacobian of output, shape ``(input_dim, sequence_length, head_dim)``
        o_lap_ref: Laplacian of output, shape ``(sequence_length, head_dim)``
    """

    q_idx = 0 if q_block_len is None else pl.program_id(1)
    q_block_len = q_block_len or q_x_ref.shape[0]
    kv_mask = mask_ref[:]
    k = k_x_ref[:, :]
    v = v_x_ref[:, :]
    k_lap = jnp.where(kv_mask[:, None], k_lap_ref[:, :], 0.0)
    v_lap = jnp.where(kv_mask[:, None], v_lap_ref[:, :], 0.0)

    q_slice = pl.dslice(q_idx * q_block_len, q_block_len)
    q_mask = pl.load(mask_ref, (q_slice,))
    square_mask = q_mask[:, None] * kv_mask[None, :]
    # Forward pass
    q = q_x_ref[:, :]
    s = jnp.where(square_mask, pl.dot(q, k, trans_b=True), -big_number(q.dtype))
    p = jax.nn.softmax(s, axis=1)
    o = pl.dot(p, v)
    o_x_ref[:, :] = o

    # Laplacian L(h) J(F) terms
    # We don't need to mask q_lap, no cross-electron contributions
    q_lap = jnp.where(q_mask[:, None], q_lap_ref[:, :], 0.0)
    qr2_k = pl.dot(q_lap, k, trans_b=True)
    qr2_k_p = qr2_k * p
    q_kr2 = pl.dot(q, k_lap, trans_b=True)
    q_kr2_p = q_kr2 * p
    o_lap = pl.dot(qr2_k_p + q_kr2_p, v)  # QR^2 OQ first term and KR^2 OK first term
    o_lap -= (
        sum_columns(qr2_k_p + q_kr2_p) * o
    )  # QR^2 OQ second term and KR^2 OK second term
    o_lap += pl.dot(p, v_lap)  ## VR^2 OV

    def body_of_loop_over_elec_coords(p_idx, o_lap):
        # Jacobian
        # We don't need to mask the electron coordinate axis of the Jacobian, no cross-electron-coordinate contributions
        q_jac = jnp.where(q_mask[:, None], q_jac_ref[p_idx, :, :], 0.0)
        k_jac = jnp.where(kv_mask[:, None], k_jac_ref[p_idx, :, :], 0.0)
        v_jac = jnp.where(kv_mask[:, None], v_jac_ref[p_idx, :, :], 0.0)
        input_mask = input_mask_ref[p_idx]

        # Jacobian
        qr_k = pl.dot(q_jac, k, trans_b=True)
        q_kr = pl.dot(q, k_jac, trans_b=True)
        qr_k_q_kr_p = (qr_k + q_kr) * p
        ## First and third terms
        o_jac = pl.dot(qr_k_q_kr_p, v)
        ## Second term and fourth terms
        qr_k_q_kr_p_sum = sum_columns(qr_k_q_kr_p)
        o_jac -= qr_k_q_kr_p_sum * o
        ## Fifth term
        o_jac += pl.dot(p, v_jac)
        o_jac_ref[p_idx, :, :] = o_jac

        # Laplacian J(h) H(F) J(h) terms
        ## Multiplies v
        qr_k_p = qr_k * p
        qr_k_p_sum = sum_columns(qr_k_p)
        q_kr_p = q_kr * p
        q_kr_p_sum = sum_columns(q_kr_p)
        qr_kr_p = pl.dot(q_jac, k_jac, trans_b=True) * p
        multiplies_v = (
            qr_k_p * qr_k  # QR QR OQQ first term
            + q_kr_p * q_kr  # KR KR OKK first term
            + 2
            * (
                -qr_k_p * qr_k_p_sum  # QR QR OQQ second and third terms
                - q_kr_p * q_kr_p_sum  # KR KR OKK second and third terms
                + qr_kr_p  # QR KR OQK first term
                + q_kr_p * qr_k  # QR KR OQK third term
                - q_kr_p_sum * qr_k_p  # QR KR OQK fourth term
                - qr_k_p_sum * q_kr_p  # QR KR OQK fifth term
                + (qr_k_p_sum * q_kr_p_sum) * p  # QR KR OQK sixth term
            )
        )
        o_lap_out = pl.dot(multiplies_v, v)
        ## Multiplies o
        multiplies_o = (
            -sum_columns(qr_k**2 * p)  # QR QR OQQ fifth term
            - sum_columns(q_kr**2 * p)  # KR KR OKK fifth term
            + 2
            * (
                +(qr_k_p_sum**2)  # QR QR OQQ fourth and sixth terms
                + q_kr_p_sum**2  # KR KR OKK fourth term
                - sum_columns(qr_kr_p)  # QR KR OQK second term
                - sum_columns(qr_k_p * q_kr)  # QR KR OQK seventh term
                + qr_k_p_sum * q_kr_p_sum  # QR KR OQK eigth term
            )
        )
        o_lap_out += multiplies_o * o
        # VR KR OVK first term and VR QR OVQ first term
        o_lap_out += pl.dot(2 * (q_kr_p + qr_k_p), v_jac)
        ## VR KR OVK second term and VR QR OVQ second term
        p_vr = pl.dot(p, v_jac)
        o_lap_out -= 2 * (q_kr_p_sum + qr_k_p_sum) * p_vr

        return o_lap + input_mask * o_lap_out

    o_lap = jax.lax.fori_loop(
        0, o_jac_ref.shape[0], body_of_loop_over_elec_coords, o_lap
    )

    o_lap_ref[:, :] = o_lap


###########################################################
# Multi head self edge attention
###########################################################


def mhsea_forward_laplacian(
    args,
    kwargs: Dict[str, Any],
    sparsity_threshold: int,
):
    del sparsity_threshold
    q, k, e, v, mask, input_mask = args
    input_len, batch_len, seq_len, num_heads, head_len = q.jacobian.dense_array.shape

    kernel = kwargs.get('kernel', 'pallas')
    interpret = kwargs.get('interpret', False)
    q_block_len = kwargs.get('q_block_len', 16)
    num_warps = kwargs.get('num_warps', seq_len // 4)
    num_stages = kwargs.get('num_stages', 1)

    q_block_len, kv_block_len = compute_q_and_kv_block_len(seq_len, q_block_len)
    v_dim = v.shape[-1]

    if kernel == 'reference':
        kernel_fn = reference_mhsea_forward_laplacian_kernel
    elif kernel == 'folx':
        kernel_fn = folx_mhsea_forward_laplacian_kernel
    elif kernel == 'pallas':
        kernel_fn = pl.pallas_call(
            partial(mhsea_forward_laplacian_kernel, q_block_len=q_block_len),
            grid=create_grid(batch_len, seq_len, num_heads, q_block_len),
            in_specs=[
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, q_block_len, True
                ),  # q.x
                get_jacobian_block_spec(
                    input_len, seq_len, head_len, q_block_len, True
                ),  # q.jacobian
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, q_block_len, True
                ),  # q.laplacian
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, kv_block_len
                ),  # k.x
                get_jacobian_block_spec(
                    input_len, seq_len, head_len, kv_block_len
                ),  # k.jacobian
                get_value_or_laplacian_block_spec(
                    seq_len, head_len, kv_block_len
                ),  # k.laplacian
                get_value_or_laplacian_block_spec(
                    seq_len, seq_len, q_block_len, True
                ),  # e.x
                get_jacobian_block_spec(
                    input_len, seq_len, seq_len, q_block_len, True
                ),  # e.jacobian
                get_value_or_laplacian_block_spec(
                    seq_len, seq_len, q_block_len, True
                ),  # e.laplacian
                get_value_or_laplacian_block_spec(seq_len, v_dim, kv_block_len),  # v.x
                get_jacobian_block_spec(
                    input_len, seq_len, v_dim, kv_block_len
                ),  # v.jacobian
                get_value_or_laplacian_block_spec(
                    seq_len, v_dim, kv_block_len
                ),  # v.laplacian
                get_mask_block_spec(seq_len, q_block_len),  # mask
                get_input_mask_block_spec(input_len, q_block_len),  # input_mask
            ],
            out_specs=[
                get_value_or_laplacian_block_spec(
                    seq_len, v_dim, q_block_len, True
                ),  # o.x
                get_jacobian_block_spec(
                    input_len, seq_len, v_dim, q_block_len, True
                ),  # o.jacobian
                get_value_or_laplacian_block_spec(
                    seq_len, v_dim, q_block_len, True
                ),  # o.laplacian
            ],
            out_shape=[
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, v_dim),
                    dtype=v.dtype,  # o.x
                ),
                jax.ShapeDtypeStruct(
                    shape=(
                        input_len,
                        batch_len,
                        seq_len,
                        num_heads,
                        v_dim,
                    ),  # o.jacobian
                    dtype=v.dtype,
                ),
                jax.ShapeDtypeStruct(
                    shape=(batch_len, seq_len, num_heads, v_dim),
                    dtype=v.dtype,  # o.laplacian
                ),
            ],
            compiler_params=dict(
                triton=dict(num_warps=num_warps, num_stages=num_stages)
            ),
            debug=False,
            interpret=interpret,
            name='mhsea_forward_laplacian',
        )
    else:
        raise ValueError(f'Unknown kernel {kernel}.')
    o, o_jacobian, o_laplacian = kernel_fn(
        q.x,
        q.jacobian.dense_array,
        q.laplacian,
        k.x,
        k.jacobian.dense_array,
        k.laplacian,
        e.x,
        e.jacobian.dense_array,
        e.laplacian,
        v.x,
        v.jacobian.dense_array,
        v.laplacian,
        mask,
        input_mask,
    )
    return FwdLaplArray(o, FwdJacobian(o_jacobian, None), o_laplacian)


@jax.jit
def reference_mhsea_forward_laplacian_kernel(
    q: jax.Array,
    q_jac: jax.Array,
    q_lap: jax.Array,
    k: jax.Array,
    k_jac: jax.Array,
    k_lap: jax.Array,
    e: jax.Array,
    e_jac: jax.Array,
    e_lap: jax.Array,
    v: jax.Array,
    v_jac: jax.Array,
    v_lap: jax.Array,
    mask: jax.Array,
    input_mask: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""Reference jax implementation of the multi-head self edge attention forward Laplacian kernel."""
    # [batch_size, seq_len, num_heads, seq_len]
    square_mask = mask[:, None, None, :] * mask[:, :, None, None]
    # [input_dim, batch_size, seq_len, num_heads, head_dim]
    coordinate_and_electron_mask = (
        input_mask[:, :, None, None, None] * mask[None, :, :, None, None]
    )
    # [batch_size, seq_len, num_heads, head_dim]
    qkv_mask = mask[:, :, None, None]
    q = jnp.where(qkv_mask, q, 0.0)
    k = jnp.where(qkv_mask, k, 0.0)
    v = jnp.where(qkv_mask, v, 0.0)
    q_jac = jnp.where(coordinate_and_electron_mask, q_jac, 0.0)
    k_jac = jnp.where(coordinate_and_electron_mask, k_jac, 0.0)
    e_jac = jnp.where(input_mask[:, :, None, None, None], e_jac, 0.0)
    v_jac = jnp.where(coordinate_and_electron_mask, v_jac, 0.0)
    q_lap = jnp.where(qkv_mask, q_lap, 0.0)
    k_lap = jnp.where(qkv_mask, k_lap, 0.0)
    v_lap = jnp.where(qkv_mask, v_lap, 0.0)

    ###########################################################################
    # Forward computation
    ###########################################################################

    square_mask = mask[:, None, None, :] * mask[:, :, None, None]
    s = jnp.einsum('Biha,Bjha->Bihj', q, k) + e
    s = jnp.where(square_mask, s, -big_number(s.dtype))
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('BnhN,BNhd->Bnhd', p, v)

    ###########################################################################
    # Forward gradient/Jacobian
    ###########################################################################
    s_jac = (
        jnp.einsum('Biha,RBjha->RBihj', q, k_jac)
        + jnp.einsum('RBiha,Bjha->RBihj', q_jac, k)
        + e_jac
    ) * square_mask
    p_jac = p * (s_jac - jnp.einsum('Bihk,RBihk->RBih', p, s_jac)[..., None])
    o_jac = jnp.einsum('RBjha,Bihj->RBiha', v_jac, p) + jnp.einsum(
        'RBihj,Bjha->RBiha', p_jac, v
    )

    ###########################################################################
    # Direct Laplacian part of the Laplacian
    ###########################################################################
    s_lap_d = (
        jnp.einsum('Biha,Bjha->Bihj', q, k_lap)
        + jnp.einsum('Biha,Bjha->Bihj', q_lap, k)
        + e_lap
    ) * square_mask
    p_lap_d = p * (s_lap_d - jnp.einsum('Bihk,Bihk->Bih', p, s_lap_d)[..., None])
    o_lap_d = jnp.einsum('Bjha,Bihj->Biha', v_lap, p) + jnp.einsum(
        'Bihj,Bjha->Biha', p_lap_d, v
    )

    ###########################################################################
    # Jacobian part of the Laplacian
    ###########################################################################
    s_lap_i = square_mask * 2 * jnp.einsum('RBiha,RBjha->Bihj', q_jac, k_jac)
    p_lap_i = (
        p_jac * (s_jac - jnp.einsum('Bihk,RBihk->RBih', p, s_jac)[..., None])
    ).sum(0) + p * (
        s_lap_i
        - jnp.einsum('RBihk,RBihk->Bih', p_jac, s_jac)[..., None]
        - jnp.einsum('Bihk,Bihk->Bih', p, s_lap_i)[..., None]
    )
    o_lap_i = jnp.einsum('Bihj,Bjha->Biha', p_lap_i, v) + 2 * jnp.einsum(
        'RBihj,RBjha->Biha', p_jac, v_jac
    )
    o_laplacian = o_lap_d + o_lap_i
    return o, o_jac, o_laplacian


@jax.jit
def folx_mhsea_forward_laplacian_kernel(
    q: jax.Array,
    q_jac: jax.Array,
    q_lap: jax.Array,
    k: jax.Array,
    k_jac: jax.Array,
    k_lap: jax.Array,
    e: jax.Array,
    e_jac: jax.Array,
    e_lap: jax.Array,
    v: jax.Array,
    v_jac: jax.Array,
    v_lap: jax.Array,
    mask: jax.Array,
    input_mask: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""Vanilla folx implementation of the multi-head self edge attention forward Laplacian kernel."""
    q_fwd_lap = FwdLaplArray(
        q, FwdJacobian.from_dense(q_jac * input_mask[..., None, None, None]), q_lap
    )
    k_fwd_lap = FwdLaplArray(
        k, FwdJacobian.from_dense(k_jac * input_mask[..., None, None, None]), k_lap
    )
    e_fwd_lap = FwdLaplArray(
        e, FwdJacobian.from_dense(e_jac * input_mask[..., None, None, None]), e_lap
    )
    v_fwd_lap = FwdLaplArray(
        v, FwdJacobian.from_dense(v_jac * input_mask[..., None, None, None]), v_lap
    )

    def fn(q, k, e, v, mask):
        return reference_mhsea_kernel(q, k, v, mask, e)

    o_fwd_lap = forward_laplacian(fn)(q_fwd_lap, k_fwd_lap, e_fwd_lap, v_fwd_lap, mask)  # type: ignore
    return o_fwd_lap.x, o_fwd_lap.jacobian.dense_array, o_fwd_lap.laplacian


def mhsea_forward_laplacian_kernel(
    q_x_ref,  # Inputs
    q_jac_ref,
    q_lap_ref,
    k_x_ref,
    k_jac_ref,
    k_lap_ref,
    e_x_ref,
    e_jac_ref,
    e_lap_ref,
    v_x_ref,
    v_jac_ref,
    v_lap_ref,
    mask_ref,
    input_mask_ref,
    o_x_ref,  # Outputs
    o_jac_ref,
    o_lap_ref,
    q_block_len: int | None,
):
    r"""The pallas implementation of the multi-head self edge attention forward Laplacian Kernel.

    Here pallas grid has already removed the batch and head dimensions.

    Args:
        q_x_ref: Queries, shape ``(sequence_length, head_dim)``
        q_jac_ref: Jacobian of queries, shape ``(input_dim, sequence_length, head_dim)``
        q_lap_ref: Laplacian of queries, shape ``(sequence_length, head_dim)``
        k_x_ref: Keys, shape ``(sequence_length, head_dim)``
        k_jac_ref: Jacobian of keys, shape ``(input_dim, sequence_length, head_dim)``
        k_lap_ref: Laplacian of keys, shape ``(sequence_length, head_dim)``
        e_x_ref: Keys, shape ``(sequence_length, sequence_length)``
        e_jac_ref: Jacobian of keys, shape ``(input_dim, sequence_length, sequence_length)``
        e_lap_ref: Laplacian of keys, shape ``(sequence_length, sequence_length)``
        v_x_ref: Values, shape ``(sequence_length, head_dim)``
        v_jac_ref: Jacobian of values, shape ``(input_dim, sequence_length, head_dim)``
        v_lap_ref: Laplacian of values, shape ``(sequence_length, head_dim)``
        mask_ref: Mask of the q, k, v values, shape ``(sequence_length,)``
        input_mask_ref: Mask of the original inputs to the model, shape ``(input_dim,)``
        o_x_ref: Output, shape ``(sequence_length, head_dim)``
        o_jac_ref: Jacobian of output, shape ``(input_dim, sequence_length, head_dim)``
        o_lap_ref: Laplacian of output, shape ``(sequence_length, head_dim)``
    """

    q_idx = 0 if q_block_len is None else pl.program_id(1)
    q_block_len = q_block_len or q_x_ref.shape[0]
    kv_mask = mask_ref[:]
    k = jnp.where(kv_mask[:, None], k_x_ref[:, :], 0.0)
    v = jnp.where(kv_mask[:, None], v_x_ref[:, :], 0.0)

    q_slice = pl.Slice(q_idx * q_block_len, q_block_len)
    q_mask = pl.load(mask_ref, (q_slice,))
    square_mask = q_mask[:, None] * kv_mask[None, :]
    # Forward pass
    q = jnp.where(q_mask[:, None], q_x_ref[:, :], 0.0)
    e = e_x_ref[:, :]
    s = jnp.where(square_mask, pl.dot(q, k, trans_b=True) + e, -big_number(q.dtype))
    p = jax.nn.softmax(s, axis=1)
    o = pl.dot(p, v)
    o_x_ref[:, :] = o

    q_lap = jnp.where(q_mask[:, None], q_lap_ref[:, :], 0.0)
    k_lap = jnp.where(kv_mask[:, None], k_lap_ref[:, :], 0.0)
    v_lap = jnp.where(kv_mask[:, None], v_lap_ref[:, :], 0.0)

    ###########################################################################
    # Direct Laplacian
    ###########################################################################
    s_lap_d = (
        pl.dot(q, k_lap, trans_b=True)
        + pl.dot(q_lap, k, trans_b=True)
        + e_lap_ref[:, :]
    ) * square_mask
    p_lap_d = p * (s_lap_d - (p * s_lap_d).sum(-1, keepdims=True))
    o_lap_d = pl.dot(p, v_lap) + pl.dot(p_lap_d, v)

    def _inner_loop(p_idx, o_lap):
        input_mask = input_mask_ref[p_idx]

        ###########################################################################
        # Jacobian
        ###########################################################################
        q_jac = jnp.where(q_mask[:, None], q_jac_ref[p_idx, :, :], 0.0)
        k_jac = jnp.where(kv_mask[:, None], k_jac_ref[p_idx, :, :], 0.0)
        v_jac = jnp.where(kv_mask[:, None], v_jac_ref[p_idx, :, :], 0.0)
        e_jac = e_jac_ref[p_idx, :, :]

        s_jac = (
            pl.dot(q, k_jac, trans_b=True) + pl.dot(q_jac, k, trans_b=True) + e_jac
        ) * square_mask
        p_jac = p * (s_jac - (p * s_jac).sum(-1, keepdims=True))
        o_jac = pl.dot(p, v_jac) + pl.dot(p_jac, v)
        o_jac_ref[p_idx, :, :] = o_jac

        ###########################################################################
        # Laplacian terms from Jacobian
        ###########################################################################
        s_lap_i = square_mask * 2 * pl.dot(q_jac, k_jac, trans_b=True)
        p_lap_i = p_jac * (s_jac - (p * s_jac).sum(-1, keepdims=True)) + p * (
            s_lap_i
            - (p_jac * s_jac).sum(-1, keepdims=True)
            - (p * s_lap_i).sum(-1, keepdims=True)
        )
        o_lap_i = pl.dot(p_lap_i, v) + 2 * pl.dot(p_jac, v_jac)

        return o_lap + input_mask * o_lap_i

    o_lap = jax.lax.fori_loop(0, o_jac_ref.shape[0], _inner_loop, o_lap_d)

    o_lap_ref[:, :] = o_lap
