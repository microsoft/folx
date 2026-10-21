"""Sparse contractions must preserve singleton batch axes during indexing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from folx import forward_laplacian


@pytest.mark.parametrize('batch_shape', [(), (1,), (1, 1), (2,)])
def test_sparse_attention_batch_axes(batch_shape):
    rng = np.random.default_rng(123)
    x = jnp.asarray(rng.normal(size=(*batch_shape, 6, 3)))
    weights = jnp.asarray(rng.normal(size=(3, 8)))

    def attention(x):
        q = jnp.tanh(x @ weights).reshape(6, 2, 4)
        scores = jnp.einsum('ihd,jhd->hij', q, q)
        return (jax.nn.softmax(scores) ** 2).sum()

    def f(x):
        if not batch_shape:
            return attention(x)
        return jax.vmap(attention)(x.reshape(-1, 6, 3)).sum()

    actual = jax.jit(forward_laplacian(f, 0.5))(x)
    expected_laplacian = jax.hessian(f)(x).reshape(x.size, x.size).trace()
    np.testing.assert_allclose(actual.x, f(x), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual.jacobian.dense_array.reshape(x.shape),
        jax.grad(f)(x),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        actual.laplacian,
        expected_laplacian,
        rtol=1e-6,
        atol=1e-6,
    )
