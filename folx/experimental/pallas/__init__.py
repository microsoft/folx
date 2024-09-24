from functools import partial
from typing import Literal

import jax
from folx import register_function

from .custom_gradients import mha_backward, mha_forward
from .forward_laplacian import mha_forward_laplacian
from .mha import mha

custom_vjp_mha = jax.custom_vjp(mha, nondiff_argnums=(5, 6, 7, 8, 9))
custom_vjp_mha.defvjp(mha_forward, mha_backward)


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9))
def multi_head_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array,
    input_mask: jax.Array,
    kernel: Literal["pallas", "reference"] = "pallas",
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
        kernel (str): Default ``pallas``. The kernel to use.
            - folx: the vanilla folx kernel is used.
            - reference: the reference jax kernel is used.
            - pallas: the pallas kernel is used.
        interpret: If ``True``, the pallas kernels are executed in interpret mode,
            which allows them to be executed e.g. on a CPU (slow). Default is ``False``.
        q_block_len (int | None): If ``None``, there is no blocking of the query
          array, otherwise it's blocked into blocks of length ``q_block_len``.
          Default is ``None``.
        num_warps (int): The number of threads to execute a single instance of the
          kernel with. Default is 4.
        num_stages (int): The number of stages. Default is 2.
    """
    return custom_vjp_mha(
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


register_function("multi_head_attention", mha_forward_laplacian)


__all__ = ["multi_head_attention"]
