import unittest

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np

from folx.ad import jacfwd


class LaplacianTestCase(unittest.TestCase):
    def setUp(self) -> None:
        jax.config.update('jax_enable_x64', True)
        return super().setUp()

    def assert_allclose(self, x, y, rtol=2e-5):
        return np.testing.assert_allclose(x, y, rtol=rtol)

    @staticmethod
    def jacobian(f, x, weights=None):
        # We use forward diff here to support complex functions
        result = jax.jit(jacfwd(f))(x)
        if weights is not None:
            return result * jnp.reshape(weights, -1)
        return result

    @staticmethod
    def laplacian(f, x, weights=None):
        flat_x, unravel = jfu.ravel_pytree(x)
        if weights is not None:
            flat_weights, _ = jfu.ravel_pytree(weights)
        else:
            flat_weights = jnp.ones((1,))

        def flat_f(flat_x):
            return jfu.ravel_pytree(f(unravel(flat_x)))[0]

        def lapl_fn(x):
            # We use forward on forward here to support complex functions
            return jnp.trace(
                jacfwd(jacfwd(flat_f))(x) * flat_weights * flat_weights[:, None],
                axis1=-2,
                axis2=-1,
            )

        return jax.jit(lapl_fn)(flat_x)
