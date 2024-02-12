import unittest
import jax
import jax.numpy as jnp
import numpy as np

from folx import forward_laplacian

from laplacian_testcase import LaplacianTestCase


class TestForwardLaplacian(LaplacianTestCase):
    def test_mlp(self):
        x = np.random.normal(size=(16,))
        layers = [
            (np.random.normal(size=(16, 16)), np.random.normal(size=(16,)))
            for i in range(4)
        ]

        @jax.jit
        def mlp(x):
            for w, b in layers:
                x = jnp.tanh(jnp.dot(x, w) + b)
            return x.sum()

        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                y = jax.jit(forward_laplacian(mlp, sparsity))(x)
                self.assertEqual(y.shape, mlp(x).shape)
                self.assertEqual(y.laplacian.shape, mlp(x).shape)
                self.assertEqual(
                    y.jacobian.dense_array.shape, self.jacobian(mlp, x).shape
                )
                self.assert_allclose(y.laplacian, self.laplacian(mlp, x))
                self.assert_allclose(y.x, mlp(x))
                self.assert_allclose(y.jacobian.dense_array, self.jacobian(mlp, x))

    def test_attention(self):
        x = np.random.normal(size=(16, 4))
        layers = [np.random.normal(size=(4, 32)) for _ in range(3)]

        @jax.jit
        def attention(x):
            q, k, v = [jnp.dot(x, w) for w in layers]
            A = jax.nn.softmax(jnp.dot(q, k.T), axis=-1)
            return jnp.dot(A, v).sum()

        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                y = jax.jit(forward_laplacian(attention, sparsity))(x)
                self.assertEqual(y.shape, attention(x).shape)
                self.assertEqual(y.laplacian.shape, attention(x).shape)
                self.assertEqual(
                    y.jacobian.dense_array.shape,
                    self.jacobian(attention, x).reshape(-1).shape,
                )
                # These checks are numerically a bit more unstable so we allow higher tolerances here
                self.assert_allclose(y.x, attention(x), rtol=1e-5)
                self.assert_allclose(
                    y.jacobian.dense_array,
                    self.jacobian(attention, x).reshape(-1),
                    rtol=1e-5,
                )
                self.assert_allclose(
                    y.laplacian, self.laplacian(attention, x), rtol=1e-5
                )


if __name__ == '__main__':
    unittest.main()
