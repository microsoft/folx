import functools
import unittest

import jax
import jax.experimental
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from folx import (
    forward_laplacian,
    wrap_forward_laplacian,
    deregister_function,
    register_function,
)

jax.config.update('jax_enable_x64', True)


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
    def test_elementwise(self):
        functions = [
            jnp.sin,
            jnp.cos,
            jnp.tanh,
            jnp.exp,
            jnp.log,
            jnp.sqrt,
            jnp.square,
            jnp.abs,
        ]
        x = np.random.randn(10) ** 2
        for f in functions:
            for sparsity in [0, x.size]:
                with self.subTest(sparsity=sparsity, f=f):
                    y = forward_laplacian(f, sparsity)(x)
                    self.assertEqual(y.x.shape, x.shape, msg=f'{f}')
                    self.assertTrue(np.allclose(y.x, f(x)), msg=f'{f}')
                    self.assertTrue(
                        np.allclose(y.jacobian.dense_array, jacobian(f, x).T),
                        msg=f'{f}',
                    )
                    self.assertTrue(
                        np.allclose(y.laplacian, laplacian(f, x)), msg=f'{f}'
                    )

    def test_matmul(self):
        x = np.random.normal(size=(16,))
        w = np.random.normal(size=(16, 16))

        @jax.jit
        def f(x):
            return jnp.matmul(x, w)

        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                y = forward_laplacian(f, sparsity)(x)
                self.assertEqual(y.x.shape, f(x).shape)
                self.assertTrue(np.allclose(y.x, f(x)))
                self.assertTrue(np.allclose(y.jacobian.dense_array, jacobian(f, x).T))
                self.assertTrue(np.allclose(y.laplacian, laplacian(f, x)))

    def test_dot(self):
        a = np.random.normal(size=(16,))
        b = np.random.normal(size=(16,))

        @jax.jit
        def f(x):
            a, b = x
            return jnp.dot(a, b)

        for sparsity in [0, a.size + b.size]:
            with self.subTest(sparsity=sparsity):
                y = jax.jit(forward_laplacian(f, sparsity))((a, b))
                self.assertEqual(y.x.shape, f((a, b)).shape)
                self.assertTrue(np.allclose(y.x, f((a, b))))
                jac = jacobian(f, (a, b))
                jac = jnp.concatenate(jtu.tree_leaves(jac), axis=0)
                self.assertTrue(np.allclose(y.jacobian.dense_array, jac))
                self.assertTrue(np.allclose(y.laplacian, laplacian(f, (a, b))))

    def test_slogdet(self):
        x = np.random.normal(size=(16 * 16))

        @jax.jit
        def f(x):
            return jnp.linalg.slogdet(x.reshape(16, 16))[1]

        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                y = jax.jit(forward_laplacian(f, sparsity))(x)
                self.assertEqual(y.x.shape, f(x).shape)
                self.assertTrue(np.allclose(y.x, f(x)))
                self.assertTrue(np.allclose(y.jacobian.dense_array, jacobian(f, x).T))
                self.assertTrue(np.allclose(y.laplacian, laplacian(f, x)))

    def test_custom_hessian(self):
        x = np.random.normal(size=(16,))

        @jax.jit
        def identity(x):
            return x

        def custom_jac_hessian_jac(args, extra_args, merge, materialize_idx):
            return jtu.tree_map(lambda x: jnp.full_like(x, 10), args.x)

        def f(x):
            return identity(x)

        register_function(
            'identity',
            wrap_forward_laplacian(
                identity,
                custom_jac_hessian_jac=custom_jac_hessian_jac,
            ),
        )

        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                y = jax.jit(forward_laplacian(f, sparsity))(x)
                self.assertEqual(y.x.shape, f(x).shape)
                self.assertTrue(np.allclose(y.x, f(x)))
                self.assertTrue(np.allclose(y.jacobian.dense_array, jacobian(f, x).T))
                self.assertTrue(np.allclose(y.laplacian, 10))
        deregister_function('identity')

    def test_dtype(self):
        x = np.random.normal(size=(16,))

        def f(x, dtype):
            return jnp.astype(x, dtype)

        for dtype in [
            jnp.float16,
            jnp.float32,
            jnp.float64,
            jnp.complex64,
            jnp.complex128,
        ]:
            with self.subTest(dtype=dtype):
                y = jax.jit(forward_laplacian(functools.partial(f, dtype=dtype)))(x)
                self.assertEqual(y.x.dtype, dtype)
                self.assertEqual(y.jacobian.dense_array.dtype, dtype)
                self.assertEqual(y.laplacian.dtype, dtype)

        for dtype in [
            jnp.bool_,
            jnp.int8,
            jnp.int16,
            jnp.int32,
            jnp.int64,
            jnp.uint8,
            jnp.uint16,
            jnp.uint32,
            jnp.uint64,
        ]:
            with self.subTest(dtype=dtype):
                y = jax.jit(forward_laplacian(functools.partial(f, dtype=dtype)))(x)
                self.assertIsInstance(y, jax.Array)


if __name__ == '__main__':
    unittest.main()
