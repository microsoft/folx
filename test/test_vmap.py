import unittest

import jax
import jax.numpy as jnp

from folx import batched_vmap


class BatchedVmapTest(unittest.TestCase):
    def test_single_argument(self):
        def fn(x):
            return jnp.sin(x).sum()

        x = jax.random.normal(jax.random.PRNGKey(42), (10, 3))
        y = jax.vmap(fn)(x)

        for batch_size in [10, 5, 1]:
            with self.subTest(batch_size=batch_size):
                y_hat = batched_vmap(fn, batch_size)(x)
                self.assertEqual(y.shape, y_hat.shape)
                self.assertTrue(jnp.allclose(y, y_hat))

    def test_uneven(self):
        def fn(x):
            return jnp.sin(x).sum()

        x = jax.random.normal(jax.random.PRNGKey(42), (21, 3))
        y = jax.vmap(fn)(x)

        for batch_size in [10, 5, 1]:
            with self.subTest(batch_size=batch_size):
                y_hat = batched_vmap(fn, batch_size)(x)
                self.assertEqual(y.shape, y_hat.shape)
                self.assertTrue(jnp.allclose(y, y_hat))

    def test_multiple_arguments(self):
        def fn(x, y):
            return jnp.vdot(jnp.sin(x), jnp.cos(y))

        a = jax.random.normal(jax.random.PRNGKey(42), (10, 3))
        b = jax.random.normal(jax.random.PRNGKey(43), (10, 3))

        y = jax.vmap(fn)(a, b)
        for batch_size in [10, 5, 1]:
            with self.subTest(batch_size=batch_size):
                y_hat = batched_vmap(fn, batch_size)(a, b)
                self.assertEqual(y.shape, y_hat.shape)
                self.assertTrue(jnp.allclose(y, y_hat))

    def test_nested(self):
        def fn(x, y):
            return jnp.vdot(jnp.sin(x), jnp.cos(y))

        a = jax.random.normal(jax.random.PRNGKey(42), (10, 3))
        b = jax.random.normal(jax.random.PRNGKey(43), (10, 3))
        y = jax.vmap(jax.vmap(fn, in_axes=(0, None)), in_axes=(None, 0))(a, b)
        for batch_size in [10, 5, 1]:
            with self.subTest(batch_size=batch_size):
                y_hat = batched_vmap(
                    batched_vmap(fn, batch_size, in_axes=(0, None)),
                    batch_size,
                    in_axes=(None, 0),
                )(a, b)
                self.assertEqual(y.shape, y_hat.shape)
                self.assertTrue(jnp.allclose(y, y_hat))


if __name__ == '__main__':
    unittest.main()
