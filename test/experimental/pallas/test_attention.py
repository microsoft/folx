import math
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from folx.api import FwdJacobian, FwdLaplArray
from folx.experimental.pallas.attention import custom_vjp_mhsa, custom_vjp_mhsea
from folx.experimental.pallas.attention.forward_laplacian import (
    mhsa_forward_laplacian,
    mhsea_forward_laplacian,
)
from folx.experimental.pallas.attention.mhsa import reference_mhsa_kernel
from folx.experimental.pallas.attention.mhsea import reference_mhsea_kernel


def random_fwd_laplacian_qkv(rng, input_dim, batch_size, seq_len, num_heads, head_dim):
    def inner(rng, sigma, head_dim):
        rng_x, rng_jacobian, rng_laplacian = jax.random.split(rng, 3)
        x = sigma * jax.random.normal(
            rng_x, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32
        )
        jacobian = sigma * jax.random.normal(
            rng_jacobian,
            (input_dim, batch_size, seq_len, num_heads, head_dim),
            dtype=jnp.float32,
        )
        laplacian = sigma * jax.random.normal(
            rng_laplacian, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float32
        )
        return FwdLaplArray(x, FwdJacobian.from_dense(jacobian), laplacian)

    rng_q, rng_k, rng_v, rng_bias = jax.random.split(rng, 4)
    q, k, v, bias = map(
        inner,
        [rng_q, rng_k, rng_v, rng_bias],
        [1 / math.sqrt(head_dim), 1, 1, 1],
        [head_dim, head_dim, head_dim, seq_len],
    )
    return q, k, v, bias


def inputs_to_mhsa(
    rng: jax.Array,
    input_dim: int,
    batch_dim: int,
    sequence_dim: int,
    num_heads: int,
    head_dim: int,
    max_sequence: int,
    *,
    only_value: bool,
    with_bias: bool = False,
):
    mask = jnp.zeros(sequence_dim, dtype=bool).at[:max_sequence].set(True)[None]
    input_mask = (
        jnp.zeros(input_dim, dtype=bool).at[: 3 * max_sequence].set(True)[:, None]
    )
    q, k, v, bias = random_fwd_laplacian_qkv(
        rng, input_dim, batch_dim, sequence_dim, num_heads, head_dim
    )
    if with_bias:
        if only_value:
            q, k, v, bias = q.x, k.x, v.x, bias.x

        return q, k, v, mask, input_mask, bias

    else:
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
    'rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence',
    [
        (jax.random.PRNGKey(0), 1, 1, 1, 1, 1),
        (jax.random.PRNGKey(1), 1, 16, 4, 32, 16),
        (jax.random.PRNGKey(2), 1, 16, 4, 32, 4),
    ],
)
def test_mhsa(rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence):
    input_dim = 3 * sequence_dim
    q, k, v, mask, input_mask = inputs_to_mhsa(
        rng,
        input_dim,
        batch_dim,
        sequence_dim,
        num_heads,
        head_dim,
        max_sequence,
        only_value=True,
    )
    o_pallas = custom_vjp_mhsa(
        q, k, v, mask, input_mask, kernel='pallas', interpret=True
    )
    o_reference = custom_vjp_mhsa(
        q, k, v, mask, input_mask, kernel='reference', interpret=True
    )

    assert jnp.allclose(
        mask_array(o_pallas, mask), mask_array(o_reference, mask), atol=1e-6
    )


@pytest.mark.parametrize(
    'rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence',
    [
        (jax.random.PRNGKey(20), 1, 1, 1, 1, 1),
        (jax.random.PRNGKey(21), 1, 16, 4, 32, 16),
        (jax.random.PRNGKey(22), 1, 16, 4, 32, 4),
    ],
)
def test_mhsea(rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence):
    input_dim = 3 * sequence_dim
    q, k, v, mask, input_mask, bias = inputs_to_mhsa(
        rng,
        input_dim,
        batch_dim,
        sequence_dim,
        num_heads,
        head_dim,
        max_sequence,
        only_value=True,
        with_bias=True,
    )
    o_pallas = custom_vjp_mhsea(
        q, k, bias, v, mask, input_mask, kernel='pallas', interpret=True
    )
    o_reference = custom_vjp_mhsea(
        q, k, bias, v, mask, input_mask, kernel='reference', interpret=True
    )

    assert jnp.allclose(
        mask_array(o_pallas, mask), mask_array(o_reference, mask), atol=1e-6
    )


@pytest.mark.parametrize(
    'rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence, with_vmap',
    [
        (jax.random.PRNGKey(3), 1, 1, 1, 1, 1, False),
        (jax.random.PRNGKey(4), 1, 16, 4, 32, 16, False),
        (jax.random.PRNGKey(5), 1, 16, 4, 32, 4, False),
        (jax.random.PRNGKey(6), 1, 1, 1, 1, 1, True),
        (jax.random.PRNGKey(7), 1, 16, 4, 32, 16, True),
        (jax.random.PRNGKey(8), 1, 16, 4, 32, 4, True),
    ],
)
def test_vjp(
    rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence, with_vmap
):
    input_dim = 3 * sequence_dim
    q, k, v, mask, input_mask = inputs_to_mhsa(
        rng,
        input_dim,
        batch_dim,
        sequence_dim,
        num_heads,
        head_dim,
        max_sequence,
        only_value=True,
    )
    if with_vmap:
        q, k, v = jax.tree.map(lambda x: x[None], (q, k, v))
    o_vjp = q

    fn = partial(
        custom_vjp_mhsa,
        mask=mask,
        input_mask=input_mask,
        kernel='pallas',
        interpret=True,
    )
    if with_vmap:
        fn = jax.vmap(fn)
    o, mhsa_vjp_fn = jax.vjp(fn, q, k, v)
    q_vjp, k_vjp, v_vjp = mhsa_vjp_fn(o_vjp)

    ref_fn = partial(
        custom_vjp_mhsa,
        mask=mask,
        input_mask=input_mask,
        kernel='reference',
        interpret=True,
    )
    if with_vmap:
        ref_fn = jax.vmap(ref_fn)
    ref_o, ref_mhsa_vjp_fn = jax.vjp(ref_fn, q, k, v)
    ref_q_vjp, ref_k_vjp, ref_v_vjp = ref_mhsa_vjp_fn(o_vjp)

    jax_fn = partial(reference_mhsa_kernel, mask=mask)
    if with_vmap:
        jax_fn = jax.vmap(jax_fn)
    jax_o, jax_mhsa_vjp_fn = jax.vjp(jax_fn, q, k, v)
    jax_q_vjp, jax_k_vjp, jax_v_vjp = jax_mhsa_vjp_fn(o_vjp)

    assert jnp.allclose(mask_array(o, mask), mask_array(ref_o, mask), atol=1e-6)
    assert jnp.allclose(mask_array(q_vjp, mask), mask_array(ref_q_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(k_vjp, mask), mask_array(ref_k_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(v_vjp, mask), mask_array(ref_v_vjp, mask), atol=1e-6)

    assert jnp.allclose(mask_array(o, mask), mask_array(jax_o, mask), atol=1e-6)
    assert jnp.allclose(mask_array(q_vjp, mask), mask_array(jax_q_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(k_vjp, mask), mask_array(jax_k_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(v_vjp, mask), mask_array(jax_v_vjp, mask), atol=1e-6)

    assert jnp.allclose(mask_array(ref_o, mask), mask_array(jax_o, mask), atol=1e-6)
    assert jnp.allclose(
        mask_array(ref_q_vjp, mask), mask_array(jax_q_vjp, mask), atol=1e-6
    )
    assert jnp.allclose(
        mask_array(ref_k_vjp, mask), mask_array(jax_k_vjp, mask), atol=1e-6
    )
    assert jnp.allclose(
        mask_array(ref_v_vjp, mask), mask_array(jax_v_vjp, mask), atol=1e-6
    )


@pytest.mark.parametrize(
    'rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence, with_vmap',
    [
        (jax.random.PRNGKey(23), 1, 1, 1, 1, 1, False),
        (jax.random.PRNGKey(24), 1, 16, 4, 32, 16, False),
        (jax.random.PRNGKey(25), 1, 16, 4, 32, 4, False),
        (jax.random.PRNGKey(26), 1, 1, 1, 1, 1, True),
        (jax.random.PRNGKey(27), 1, 16, 4, 32, 16, True),
        (jax.random.PRNGKey(28), 1, 16, 4, 32, 4, True),
    ],
)
def test_mhsea_vjp(
    rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence, with_vmap
):
    input_dim = 3 * sequence_dim
    q, k, v, mask, input_mask, bias = inputs_to_mhsa(
        rng,
        input_dim,
        batch_dim,
        sequence_dim,
        num_heads,
        head_dim,
        max_sequence,
        only_value=True,
        with_bias=True,
    )
    if with_vmap:
        q, k, v, bias = jax.tree.map(lambda x: x[None], (q, k, v, bias))
    o_vjp = jnp.where(
        mask[:, :, None, None], q, 0.0
    )  # Without this, v_vjp can be wrong

    fn = partial(
        custom_vjp_mhsea,
        mask=mask,
        input_mask=input_mask,
        kernel='pallas',
        interpret=True,
    )
    if with_vmap:
        fn = jax.vmap(fn)
    o, mhsea_vjp_fn = jax.vjp(fn, q, k, bias, v)
    q_vjp, k_vjp, bias_vjp, v_vjp = mhsea_vjp_fn(o_vjp)

    ref_fn = partial(
        custom_vjp_mhsea,
        mask=mask,
        input_mask=input_mask,
        kernel='reference',
        interpret=True,
    )
    if with_vmap:
        ref_fn = jax.vmap(ref_fn)
    ref_o, ref_mhsea_vjp_fn = jax.vjp(ref_fn, q, k, bias, v)
    ref_q_vjp, ref_k_vjp, ref_bias_vjp, ref_v_vjp = ref_mhsea_vjp_fn(o_vjp)

    def jax_fn(q, k, bias, v):
        return reference_mhsea_kernel(q, k, v, mask=mask, edges=bias)

    if with_vmap:
        jax_fn = jax.vmap(jax_fn)
    jax_o, jax_mhsa_vjp_fn = jax.vjp(jax_fn, q, k, bias, v)
    jax_q_vjp, jax_k_vjp, jax_bias_vjp, jax_v_vjp = jax_mhsa_vjp_fn(o_vjp)

    assert jnp.allclose(mask_array(o, mask), mask_array(ref_o, mask), atol=1e-6)
    assert jnp.allclose(mask_array(q_vjp, mask), mask_array(ref_q_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(k_vjp, mask), mask_array(ref_k_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(v_vjp, mask), mask_array(ref_v_vjp, mask), atol=1e-6)

    assert jnp.allclose(mask_array(o, mask), mask_array(jax_o, mask), atol=1e-6)
    assert jnp.allclose(mask_array(q_vjp, mask), mask_array(jax_q_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(k_vjp, mask), mask_array(jax_k_vjp, mask), atol=1e-6)
    assert jnp.allclose(mask_array(v_vjp, mask), mask_array(jax_v_vjp, mask), atol=1e-6)

    assert jnp.allclose(mask_array(ref_o, mask), mask_array(jax_o, mask), atol=1e-6)
    assert jnp.allclose(
        mask_array(ref_q_vjp, mask), mask_array(jax_q_vjp, mask), atol=1e-6
    )
    assert jnp.allclose(
        mask_array(ref_k_vjp, mask), mask_array(jax_k_vjp, mask), atol=1e-6
    )
    assert jnp.allclose(
        mask_array(ref_v_vjp, mask), mask_array(jax_v_vjp, mask), atol=1e-6
    )


@pytest.mark.parametrize(
    'rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence',
    [
        (jax.random.PRNGKey(0), 1, 1, 1, 1, 1),
        (jax.random.PRNGKey(1), 1, 16, 4, 32, 16),
        (jax.random.PRNGKey(2), 1, 16, 4, 32, 4),
    ],
)
def test_forward_laplacian(
    rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence
):
    input_dim = 3 * sequence_dim
    q, k, v, mask, input_mask = inputs_to_mhsa(
        rng,
        input_dim,
        batch_dim,
        sequence_dim,
        num_heads,
        head_dim,
        max_sequence,
        only_value=False,
    )

    folx_out = mask_fwd_lapl_array(
        mhsa_forward_laplacian(
            (q, k, v, mask, input_mask), {'kernel': 'folx', 'interpret': True}, 0
        ),
        mask,
        input_mask,
    )
    ref_out = mask_fwd_lapl_array(
        mhsa_forward_laplacian(
            (q, k, v, mask, input_mask), {'kernel': 'reference', 'interpret': True}, 0
        ),
        mask,
        input_mask,
    )
    out = mask_fwd_lapl_array(
        mhsa_forward_laplacian(
            (q, k, v, mask, input_mask), {'kernel': 'pallas', 'interpret': True}, 0
        ),
        mask,
        input_mask,
    )

    assert jnp.allclose(folx_out.x, ref_out.x, atol=1e-6)
    assert jnp.allclose(
        folx_out.jacobian.dense_array, ref_out.jacobian.dense_array, atol=1e-6
    )
    assert jnp.allclose(folx_out.laplacian, ref_out.laplacian, atol=5e-5)

    assert jnp.allclose(folx_out.x, out.x, atol=1e-6)
    assert jnp.allclose(
        folx_out.jacobian.dense_array, out.jacobian.dense_array, atol=1e-6
    )
    assert jnp.allclose(folx_out.laplacian, out.laplacian, atol=5e-5)

    assert jnp.allclose(ref_out.x, out.x, atol=1e-6)
    assert jnp.allclose(
        ref_out.jacobian.dense_array, out.jacobian.dense_array, atol=1e-6
    )
    assert jnp.allclose(ref_out.laplacian, out.laplacian, atol=5e-5)


@pytest.mark.parametrize(
    'rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence',
    [
        (jax.random.PRNGKey(30), 1, 1, 1, 1, 1),
        (jax.random.PRNGKey(31), 1, 16, 4, 32, 16),
        (jax.random.PRNGKey(32), 1, 16, 4, 32, 4),
    ],
)
def test_mhsea_forward_laplacian(
    rng, batch_dim, sequence_dim, num_heads, head_dim, max_sequence
):
    input_dim = 3 * sequence_dim
    q, k, v, mask, input_mask, bias = inputs_to_mhsa(
        rng,
        input_dim,
        batch_dim,
        sequence_dim,
        num_heads,
        head_dim,
        max_sequence,
        only_value=False,
        with_bias=True,
    )

    folx_out = mask_fwd_lapl_array(
        mhsea_forward_laplacian(
            (q, k, bias, v, mask, input_mask), {'kernel': 'folx', 'interpret': True}, 0
        ),
        mask,
        input_mask,
    )
    ref_out = mask_fwd_lapl_array(
        mhsea_forward_laplacian(
            (q, k, bias, v, mask, input_mask),
            {'kernel': 'reference', 'interpret': True},
            0,
        ),
        mask,
        input_mask,
    )
    out = mask_fwd_lapl_array(
        mhsea_forward_laplacian(
            (q, k, bias, v, mask, input_mask),
            {'kernel': 'pallas', 'interpret': True},
            0,
        ),
        mask,
        input_mask,
    )

    assert jnp.allclose(folx_out.x, ref_out.x, atol=1e-6)
    assert jnp.allclose(
        folx_out.jacobian.dense_array, ref_out.jacobian.dense_array, atol=1e-6
    )
    assert jnp.allclose(folx_out.laplacian, ref_out.laplacian, atol=1e-4)

    assert jnp.allclose(folx_out.x, out.x, atol=1e-6)
    assert jnp.allclose(
        folx_out.jacobian.dense_array, out.jacobian.dense_array, atol=1e-6
    )
    assert jnp.allclose(folx_out.laplacian, out.laplacian, atol=1e-4)

    assert jnp.allclose(ref_out.x, out.x, atol=1e-6)
    assert jnp.allclose(
        ref_out.jacobian.dense_array, out.jacobian.dense_array, atol=1e-6
    )
    assert jnp.allclose(ref_out.laplacian, out.laplacian, atol=1e-4)
