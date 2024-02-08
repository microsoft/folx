import unittest

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np


class LaplacianTestCase(unittest.TestCase):
    def setUp(self) -> None:
        jax.config.update('jax_enable_x64', True)
        return super().setUp()

    def assert_allclose(self, x, y):
        return np.testing.assert_allclose(x, y)

    @staticmethod
    def jacobian(f, x):
        # We use forward diff here to support complex functions
        return jax.jit(jax.jacfwd(f))(x)

    @staticmethod
    def laplacian(f, x):
        flat_x, unravel = jfu.ravel_pytree(x)

        def flat_f(flat_x):
            return jfu.ravel_pytree(f(unravel(flat_x)))[0]

        def lapl_fn(x):
            # We use forward on forward here to support complex functions
            return jnp.trace(jax.jacfwd(jax.jacfwd(flat_f))(x), axis1=-2, axis2=-1)

        return jax.jit(lapl_fn)(flat_x)
