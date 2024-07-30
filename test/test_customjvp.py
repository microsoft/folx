import jax
import jax.numpy as jnp
import numpy as np
from laplacian_testcase import LaplacianTestCase

from folx import forward_laplacian


class TestForwardLaplacianJvp(LaplacianTestCase):
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
        self.assert_allclose(y.x, fn(x))
        self.assert_allclose(y.jacobian.dense_array, self.jacobian(fn, x).T)
        self.assert_allclose(y.laplacian, self.laplacian(fn, x))
