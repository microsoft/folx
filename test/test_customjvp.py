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


class TestForwardLaplacianJvp(unittest.TestCase):
    def test_customjvp(self):
        @jax.custom_jvp
        def fn(x):
            return jnp.sin(x)

        @fn.defjvp
        def fn_jvp(primals, tangents):
            # Custom differs by a factor of 2 to the true Jacobian
            (x,) = primals
            (x_dot,) = tangents
            return fn(x), 2 * x_dot * jnp.cos(x)

        x = np.random.randn(10)
        y = forward_laplacian(fn, 0)(x)
        self.assertEqual(y.x.shape, x.shape)
        self.assertTrue(np.allclose(y.x, fn(x)))
        self.assertTrue(np.allclose(y.jacobian.dense_array, jacobian(fn, x).T))
        self.assertTrue(np.allclose(y.laplacian, laplacian(fn, x)))
