import jax

from .custom_gradients import mha_backward, mha_forward
from .mha import mha

custom_vjp_mha = jax.custom_vjp(mha, nondiff_argnums=(5, 6, 7, 8, 9))
custom_vjp_mha.defvjp(mha_forward, mha_backward)
