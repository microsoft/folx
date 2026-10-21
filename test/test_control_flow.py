import jax
import jax.numpy as jnp
import numpy as np
import pytest
from laplacian_testcase import LaplacianTestCase
from parameterized import parameterized

from folx import forward_laplacian


@jax.custom_vjp
def _custom_vjp_sin(x):
    return jnp.sin(x)


_custom_vjp_sin.defvjp(lambda x: (jnp.sin(x), x), lambda x, g: (g * jnp.cos(x),))


def _scan(x):
    w = jnp.arange(1.0, 4.0)
    c, ys = jax.lax.scan(lambda c, w: (jnp.tanh(c * w), c**2), x, w)
    return jnp.concatenate([c, ys.ravel()])


def _scan_reverse(x):
    w = jnp.ones(2)
    c, ys = jax.lax.scan(lambda c, w: (jnp.sin(c) + w, c * w), x, w, reverse=True)
    return jnp.concatenate([c, ys.ravel()])


def _scan_promotes_carry(x):
    # The second carry only becomes input dependent inside the loop.
    def body(c, _):
        return (jnp.sin(c[0]), c[1] + c[0] ** 2), None

    return jax.lax.scan(body, (x, jnp.zeros_like(x)), length=3)[0][1]


def _scan_mixes_carry(x):
    # Rolling mixes the per-element sparsity of the carry.
    return jax.lax.scan(lambda c, _: (jnp.roll(c, 1) * c, None), x, length=2)[0]


CASES = [
    ('scan', _scan),
    ('scan_reverse', _scan_reverse),
    ('scan_promotes_carry', _scan_promotes_carry),
    ('scan_mixes_carry', _scan_mixes_carry),
    ('fori_loop', lambda x: jax.lax.fori_loop(0, 3, lambda i, c: jnp.sin(c) * i, x)),
    (
        'while_loop',
        lambda x: jax.lax.while_loop(
            lambda c: c[1] < 3, lambda c: (jnp.sin(c[0]), c[1] + 1), (x, 0)
        )[0],
    ),
    ('cond', lambda x: jax.lax.cond(x.sum() > 0, jnp.sin, jnp.cos, x)),
    ('remat', jax.checkpoint(lambda x: jnp.sin(x) ** 2 * x)),
]


class TestControlFlow(LaplacianTestCase):
    @parameterized.expand(
        [(name, f, sparsity) for name, f in CASES for sparsity in (0, 4)]
    )
    def test_laplacian(self, name, f, sparsity):
        x = np.random.default_rng(0).normal(size=4)
        y = forward_laplacian(f, sparsity)(x)
        self.assert_allclose(y.x, f(x))
        self.assert_allclose(y.jacobian.dense_array, self.jacobian(f, x).T)
        self.assert_allclose(y.laplacian, self.laplacian(f, x))

    @parameterized.expand([(0,), (4,)])
    def test_custom_vjp(self, sparsity):
        x = np.random.default_rng(0).normal(size=4)
        y = forward_laplacian(_custom_vjp_sin, sparsity)(x)
        self.assert_allclose(y.x, np.sin(x))
        self.assert_allclose(y.jacobian.dense_array, np.diag(np.cos(x)))
        self.assert_allclose(y.laplacian, -np.sin(x))

    def test_scan_keeps_sparsity(self):
        y = forward_laplacian(_scan, 4)(np.ones(4))
        assert y.jacobian.data.shape[0] == 1

    def test_scan_over_input_dependent_xs_raises(self):
        def f(x):
            return jax.lax.scan(lambda c, a: (c + a, None), 0.0, x)[0]

        with pytest.raises(NotImplementedError, match='unroll'):
            forward_laplacian(f)(np.ones(4))

    def test_while_condition_on_input_raises(self):
        def f(x):
            return jax.lax.while_loop(lambda c: c < x.sum(), lambda c: c + 1.0, 0.0)

        with pytest.raises(NotImplementedError, match='condition'):
            forward_laplacian(f)(np.ones(4))
