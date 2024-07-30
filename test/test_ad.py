import unittest

import jax
import jax.numpy as jnp
import numpy as np
from laplacian_testcase import LaplacianTestCase

from folx.ad import hessian, jacfwd, jacrev


class TestAutoDiff(LaplacianTestCase):
    def test_jacfwd(self):
        def f(x):
            return jnp.sin(x * x.sum())

        x = np.random.normal(size=(3, 4))
        target_shape = (*x.shape, x.size)
        self.assertEqual(jacfwd(f)(x).shape, target_shape)
        self.assert_allclose(jacfwd(f)(x), jax.jacfwd(f)(x).reshape(target_shape))

    def test_jacrev(self):
        def f(x):
            return jnp.sin(x * x.sum())

        x = np.random.normal(size=(3, 4))
        target_shape = (x.size, *x.shape)
        self.assertEqual(jacrev(f)(x).shape, target_shape)
        self.assert_allclose(jacrev(f)(x), jax.jacrev(f)(x).reshape(target_shape))

    def test_hessian(self):
        def f(x):
            return jnp.sin(x * x.sum())

        x = np.random.normal(size=(12))
        target_shape = (x.size, x.size, x.size)
        self.assertEqual(hessian(f)(x).shape, target_shape)
        self.assert_allclose(hessian(f)(x), jax.hessian(f)(x))


if __name__ == '__main__':
    unittest.main()
