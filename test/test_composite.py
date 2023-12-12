import unittest

import jax
import jax.experimental
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np

from folx import forward_laplacian


def jacobian(f, x):
    return jax.jit(jax.jacobian(f))(x)


def laplacian(f, x):
    flat_x, unravel = jfu.ravel_pytree(x)
    def flat_f(flat_x):
        return jfu.ravel_pytree(f(unravel(flat_x)))[0]
    def lapl_fn(x):
        return jnp.trace(jax.hessian(flat_f)(x), axis1=-2, axis2=-1)
    return jax.jit(lapl_fn)(flat_x)


class TestForwardLaplacian(unittest.TestCase):
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
                self.assertEqual(y.jacobian.dense_array.shape, jacobian(mlp, x).shape)
                self.assertTrue(np.allclose(y.laplacian, laplacian(mlp, x)))
                self.assertTrue(np.allclose(y.x, mlp(x)))
                self.assertTrue(np.allclose(y.jacobian.dense_array, jacobian(mlp, x)))

    def test_attention(self):
        x = np.random.normal(size=(16, 4))
        layers = [
            np.random.normal(size=(4, 32))
            for _ in range(3)
        ]
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
                self.assertEqual(y.jacobian.dense_array.shape, jacobian(attention, x).reshape(-1).shape)
                self.assertTrue(np.allclose(y.x, attention(x)))
                self.assertTrue(np.allclose(y.jacobian.dense_array, jacobian(attention, x).reshape(-1)))
                self.assertTrue(np.allclose(y.laplacian, laplacian(attention, x)))
