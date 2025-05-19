from typing import Literal

import jax

from ....wrapped_functions import register_function
from .custom_gradients import mhsa_backward, mhsa_forward, mhsea_backward, mhsea_forward
from .forward_laplacian import mhsa_forward_laplacian, mhsea_forward_laplacian
from .mhsa import mhsa
from .mhsea import mhsea

custom_vjp_mhsa = jax.custom_vjp(mhsa, nondiff_argnums=(5, 6, 7, 8, 9))
custom_vjp_mhsa.defvjp(mhsa_forward, mhsa_backward)
custom_vjp_mhsa_jitted = jax.jit(custom_vjp_mhsa, static_argnums=(5, 6, 7, 8, 9))
register_function('custom_vjp_mhsa_jitted', mhsa_forward_laplacian)

custom_vjp_mhsea = jax.custom_vjp(mhsea, nondiff_argnums=(6, 7, 8, 9, 10))
custom_vjp_mhsea.defvjp(mhsea_forward, mhsea_backward)
custom_vjp_mhsea_jitted = jax.jit(custom_vjp_mhsea, static_argnums=(6, 7, 8, 9, 10))
register_function('custom_vjp_mhsea_jitted', mhsea_forward_laplacian)


def multi_head_self_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    # TODO: support multiple masks for cross-attention
    mask: jax.Array,
    input_mask: jax.Array,
    bias: jax.Array | None = None,
    *,
    kernel: Literal['pallas', 'reference'] = 'pallas',
    interpret: bool = False,
    q_block_len: int | None = None,
    num_warps: int = 2,
    num_stages: int = 2,
):
    r"""Compute multi-head attention (support VJP not JVP).

    Having this wrapper jit block is necessary for folx to recognize the attention block.

    Args:
        q: Queries of shape ``(batch_size, sequence_length, num_heads, head_dim)``
        k: Keys of shape ``(batch_size, sequence_length, num_heads, head_dim)``
        v: Values of shape ``(batch_size, sequence_length, num_heads, head_dim)``
        mask: Mask of the q, k, v values, shape ``(batch_size, sequence_length)``
        input_mask: Used only during mode forward Laplacian: mask of the original
            input to the model, with respect to which the forward Laplacian is computed.
            For us, normally of shape ``(3 * sequence_length, batch_size)``, but
            if ``q``, ``k``, ``v``  are padded (e.g. in FLASH attention below with
            ``n_elec < 16``), this should still retain the original ``3 * n_elec``
            length.
        bias: Edge bias of shape ``(batch_size, sequence_length, num_heads, sequence_length)``
            This argument is option, pass ``None`` to use no edge bias.
        kernel: Default ``pallas``. The kernel to use.
            - folx: the vanilla folx kernel is used.
            - reference: the reference jax kernel is used.
            - pallas: the pallas kernel is used.
        interpret: If ``True``, the pallas kernels are executed in interpret mode,
            which allows them to be executed e.g. on a CPU (slow). Default is ``False``.
        q_block_len (int | None): If ``None``, there is no blocking of the query
          array, otherwise it's blocked into blocks of length ``q_block_len``.
          Default is ``None``.
        num_warps: The number of threads to execute a single instance of the
          kernel with. Default is 2.
        num_stages: The number of stages. Default is 2.
    """
    if bias is None:
        return custom_vjp_mhsa_jitted(
            q,
            k,
            v,
            mask,
            input_mask,
            kernel,
            interpret,
            q_block_len,
            num_warps,
            num_stages,
        )
    else:
        return custom_vjp_mhsea_jitted(
            q,
            k,
            bias,
            v,
            mask,
            input_mask,
            kernel,
            interpret,
            q_block_len,
            num_warps,
            num_stages,
        )


__all__ = ['multi_head_self_attention']
