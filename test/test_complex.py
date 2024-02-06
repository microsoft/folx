import unittest

import jax
import jax.experimental
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np

from folx import forward_laplacian

jax.config.update('jax_enable_x64', True)


def jacobian(f, x):
    return jax.jit(jax.jacobian(f))(x)


def laplacian(f, x):
    flat_x, unravel = jfu.ravel_pytree(x)

    def flat_f(flat_x):
        return jfu.ravel_pytree(f(unravel(flat_x)))[0]

    def lapl_fn(x):
        return jnp.trace(jax.hessian(flat_f)(x), axis1=-2, axis2=-1)

    return jax.jit(lapl_fn)(flat_x)


class TestComplexFwdLapl(unittest.TestCase):
    def test_imag(self):
        x = np.random.randn(10)
        W1 = np.random.randn(10, 10)
        W2 = np.random.randn(10, 10)

        def f(x):
            return x @ W1 + 1j * x @ W2

        for g in [jnp.real, jnp.imag]:

            def h(x):
                return g(f(x))

            y = forward_laplacian(h, 0)(x)
            self.assertEqual(y.x.shape, x.shape)
            self.assertTrue(np.allclose(y.x, g(f(x))))
            self.assertTrue(np.allclose(y.jacobian.dense_array, jacobian(h, x).T))
            self.assertTrue(np.allclose(y.laplacian, laplacian(h, x)))
