import math
from functools import partial

import jax
import jax.numpy as jnp
import pytest
from folx.api import FwdJacobian, FwdLaplArray
from folx.experimental.pallas import custom_vjp_mha


def random_fwd_laplacian_qkv(rng, input_dim, batch_size, seq_len, num_heads, head_dim):
    def inner(rng, sigma):
        rng_x, rng_jacobian, rng_laplacian = jax.random.split(rng, 3)
        x = sigma * jax.random.normal(
            rng_x, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32
        )
        jacobian = sigma * jax.random.normal(
            rng_jacobian, (input_dim, batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32
        )
        laplacian = sigma * jax.random.normal(
            rng_laplacian, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32
        )
        return FwdLaplArray(x, FwdJacobian.from_dense(jacobian), laplacian)

    rng_q, rng_k, rng_v = jax.random.split(rng, 3)
    q, k, v = map(inner, [rng_q, rng_k, rng_v], [1 / math.sqrt(head_dim), 1, 1])
    return q, k, v


def inputs_to_mha(
    rng: jax.Array,
    input_dim: int,
    batch_dim: int,
    sequence_dim: int,
    num_heads: int,
    head_dim: int,
    max_sequence: int,
    only_value: bool,
):
    mask = jnp.zeros(sequence_dim, dtype=bool).at[:max_sequence].set(True)[None]
    input_mask = jnp.zeros(input_dim, dtype=bool).at[: 3 * max_sequence].set(True)[:, None]
    q, k, v = random_fwd_laplacian_qkv(rng, input_dim, batch_dim, sequence_dim, num_heads, head_dim)
    if only_value:
        q, k, v = q.x, k.x, v.x

    return q, k, v, mask, input_mask


def mask_array(array: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.where(mask[:, :, None, None], array, 0.0)


def mask_fwd_lapl_array(fwd_lap, mask, input_mask):
    jacobian_mask = input_mask[:, :, None, None, None] * mask[None, :, :, None, None]
    x = mask_array(fwd_lap.x, mask)
    laplacian = mask_array(fwd_lap.laplacian, mask)
    jacobian = jnp.where(jacobian_mask, fwd_lap.jacobian.dense_array, 0.0)
    return FwdLaplArray(x, FwdJacobian.from_dense(jacobian), laplacian)


@pytest.mark.parametrize(
    "rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence",
    [
        (jax.random.PRNGKey(0), 1, 1, 1, 1, 1),
        (jax.random.PRNGKey(1), 1, 16, 4, 32, 16),
        (jax.random.PRNGKey(2), 1, 16, 4, 32, 4),
    ],
)
def test_mha(rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence):
    input_dim = 3 * sequence_dim
    q, k, v, mask, input_mask = inputs_to_mha(
        rng, input_dim, batch_dim, sequence_dim, num_heads, head_dim, max_sequence, True
    )
    o_pallas = custom_vjp_mha(q, k, v, mask, input_mask, kernel="pallas", interpret=True)
    o_reference = custom_vjp_mha(q, k, v, mask, input_mask, kernel="reference", interpret=True)

    assert jnp.allclose(mask_array(o_pallas, mask), mask_array(o_reference, mask), atol=1e-6)


@pytest.mark.parametrize(
    "rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence, with_vmap",
    [
        (jax.random.PRNGKey(3), 1, 1, 1, 1, 1, False),
        (jax.random.PRNGKey(4), 1, 16, 4, 32, 16, False),
        (jax.random.PRNGKey(5), 1, 16, 4, 32, 4, False),
        (jax.random.PRNGKey(6), 1, 1, 1, 1, 1, True),
        (jax.random.PRNGKey(7), 1, 16, 4, 32, 16, True),
        (jax.random.PRNGKey(8), 1, 16, 4, 32, 4, True),
    ],
)
def test_vjp(rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence, with_vmap):
    input_dim = 3 * sequence_dim
    q, k, v, mask, input_mask = inputs_to_mha(
        rng, input_dim, batch_dim, sequence_dim, num_heads, head_dim, max_sequence, True
    )
    if with_vmap:
        q, k, v = jax.tree.map(lambda x: x[None], (q, k, v))
    o_vjp = q

    fn = partial(custom_vjp_mha, mask=mask, input_mask=input_mask, kernel="pallas", interpret=True)
    if with_vmap:
        fn = jax.vmap(fn)
    o, mha_vjp_fn = jax.vjp(fn, q, k, v)
    q_vjp, k_vjp, v_vjp = mha_vjp_fn(o_vjp)

    ref_fn = partial(
        custom_vjp_mha, mask=mask, input_mask=input_mask, kernel="reference", interpret=True
    )
    if with_vmap:
        ref_fn = jax.vmap(ref_fn)
    ref_o, ref_mha_vjp_fn = jax.vjp(ref_fn, q, k, v)
    ref_q_vjp, ref_k_vjp, ref_v_vjp = ref_mha_vjp_fn(o_vjp)

    jax_fn = partial(custom_vjp_mha, mask=mask)
    if with_vmap:
        jax_fn = jax.vmap(jax_fn)
    jax_o, jax_mha_vjp_fn = jax.vjp(jax_fn, q, k, v)
    jax_q_vjp, jax_k_vjp, jax_v_vjp = jax_mha_vjp_fn(o_vjp)

    assert jnp.allclose(mask_array(o, mask), mask_array(ref_o, mask), atol=1e-6)
    assert jnp.allclose(mask_array(q_vjp, mask), mask_array(ref_q_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(k_vjp, mask), mask_array(ref_k_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(v_vjp, mask), mask_array(ref_v_vjp, mask))

    assert jnp.allclose(mask_array(o, mask), mask_array(jax_o, mask), atol=1e-6)
    assert jnp.allclose(mask_array(q_vjp, mask), mask_array(jax_q_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(k_vjp, mask), mask_array(jax_k_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(v_vjp, mask), mask_array(jax_v_vjp, mask))

    assert jnp.allclose(mask_array(ref_o, mask), mask_array(jax_o, mask), atol=1e-6)
    assert jnp.allclose(mask_array(ref_q_vjp, mask), mask_array(jax_q_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(ref_k_vjp, mask), mask_array(jax_k_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(ref_v_vjp, mask), mask_array(jax_v_vjp, mask))
