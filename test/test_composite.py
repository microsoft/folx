import unittest

import jax
import jax.numpy as jnp
import numpy as np
from laplacian_testcase import LaplacianTestCase
from parameterized import parameterized

from folx import forward_laplacian


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
                gt_y, gt_jac, gt_lapl = (
                    mlp(x),
                    self.jacobian(mlp, x),
                    self.laplacian(mlp, x),
                )
                self.assertEqual(y.shape, gt_y.shape)
                self.assertEqual(y.laplacian.shape, gt_y.shape)
                self.assertEqual(y.jacobian.dense_array.shape, gt_jac.shape)
                self.assert_allclose(y.x, gt_y)
                self.assert_allclose(y.jacobian.dense_array, gt_jac)
                self.assert_allclose(y.laplacian, gt_lapl)

    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def test_attention(self, weighted: bool = False):
        x = np.random.normal(size=(16, 4))
        if weighted:
            weights = np.random.normal(size=(16, 4))
        else:
            weights = 1
        layers = [np.random.normal(size=(4, 32)) for _ in range(3)]

        @jax.jit
        def attention(x):
            q, k, v = [jnp.dot(x, w) for w in layers]
            A = jax.nn.softmax(jnp.dot(q, k.T), axis=-1)
            return jnp.dot(A, v).sum()

        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                y = jax.jit(forward_laplacian(attention, sparsity))(x, weights=weights)
                gt_y, gt_jac, gt_lapl = (
                    attention(x),
                    self.jacobian(attention, x, weights),
                    self.laplacian(attention, x, weights),
                )
                self.assertEqual(y.shape, gt_y.shape)
                self.assertEqual(y.laplacian.shape, gt_y.shape)
                gt_jac = self.jacobian(attention, x, weights)
                self.assertEqual(
                    y.jacobian.dense_array.shape,
                    gt_jac.reshape(-1).shape,
                )
                # These checks are numerically a bit more unstable so we allow higher tolerances here
                self.assert_allclose(y.x, gt_y, rtol=1e-5)
                self.assert_allclose(
                    y.jacobian.dense_array,
                    gt_jac.reshape(-1),
                    rtol=1e-5,
                )
                self.assert_allclose(y.laplacian, gt_lapl, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
