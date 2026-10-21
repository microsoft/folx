import jax
import jax.numpy as jnp
import numpy as np
from laplacian_testcase import LaplacianTestCase
from parameterized import parameterized

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

    @parameterized.expand(
        [
            ('relu', jax.nn.relu),
            ('custom_rule', None),
        ]
    )
    def test_elementwise_customjvp_keeps_sparsity(self, name, act):
        if act is None:

            @jax.custom_jvp
            def act(x):
                return jnp.sin(x)

            @act.defjvp
            def act_jvp(primals, tangents):
                # Custom differs by a factor of 2 to the true Jacobian
                return act(primals[0]), 2 * tangents[0] * jnp.cos(primals[0])

        W = np.random.default_rng(1).normal(size=(4, 3))

        def f(x):
            # Each row only depends on one input.
            return act(x[:, None] * W + 0.1)

        x = np.random.default_rng(0).normal(size=4)
        y = forward_laplacian(f, 4)(x)
        self.assertEqual(y.jacobian.data.shape[0], 1)
        dense = forward_laplacian(f, 0)(x)
        self.assert_allclose(y.jacobian.dense_array, dense.jacobian.dense_array)
        self.assert_allclose(y.laplacian, dense.laplacian)
        if name != 'custom_rule':
            self.assert_allclose(y.laplacian, self.laplacian(f, x).reshape(4, 3))

    def test_customjvp_multiple_inputs(self):
        def f(x):
            return jax.scipy.special.xlogy(x**2 + 1, x**2 + 2)

        x = np.random.default_rng(0).normal(size=4)
        y = forward_laplacian(f, 4)(x)
        self.assert_allclose(y.laplacian, self.laplacian(f, x))
