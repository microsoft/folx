import jax.numpy as jnp
import numpy as np
from laplacian_testcase import LaplacianTestCase

from folx import forward_laplacian


class TestComplexFwdLapl(LaplacianTestCase):
    def test_parts(self):
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
            self.assert_allclose(y.x, g(f(x)))
            self.assert_allclose(y.jacobian.dense_array, self.jacobian(h, x).T)
            self.assert_allclose(y.laplacian, self.laplacian(h, x))

    def test_complex_mlp(self):
        x = np.random.randn(10)
        W1 = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
        W2 = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

        def f(x):
            return jnp.tanh(x @ W1) @ W2

        y = forward_laplacian(f, 0)(x)
        self.assertEqual(y.x.shape, x.shape)
        self.assert_allclose(y.x, f(x))
        self.assert_allclose(y.jacobian.dense_array, self.jacobian(f, x).T)
        self.assert_allclose(y.laplacian, self.laplacian(f, x))

    def test_complex_abs(self):
        x = np.random.randn(10)
        W = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)

        def f(x):
            return jnp.abs(jnp.tanh(x @ W))

        y = forward_laplacian(f, 0)(x)
        self.assertEqual(y.x.shape, x.shape)
        self.assert_allclose(y.x, f(x))
        self.assert_allclose(y.jacobian.dense_array, self.jacobian(f, x).T)
        self.assert_allclose(y.laplacian, self.laplacian(f, x))
