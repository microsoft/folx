import functools

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from laplacian_testcase import LaplacianTestCase
from parameterized import parameterized

from folx import (
    deregister_function,
    forward_laplacian,
    register_function,
    wrap_forward_laplacian,
)
from folx.api import FwdLaplArray


class TestForwardLaplacian(LaplacianTestCase):
    @parameterized.expand([(False,), (True,)])
    def test_summation(self, test_complex: bool):
        x = np.random.randn(10)
        if test_complex:
            x = x + 1j * np.random.randn(10)
        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                y = forward_laplacian(jnp.sum, sparsity)(x)
                self.assertEqual(y.x.shape, ())
                self.assert_allclose(y.x, jnp.sum(x))
                self.assert_allclose(y.jacobian.dense_array, self.jacobian(jnp.sum, x))
                self.assert_allclose(y.laplacian, 0)

    @parameterized.expand([(False,), (True,)])
    def test_elementwise(self, test_complex: bool):
        functions = [
            jnp.sin,
            jnp.cos,
            jnp.tanh,
            jnp.exp,
            jnp.square,
            jnp.abs,
            # These functions only work on positive numbers
            lambda x: jnp.log(jnp.abs(x)),
            lambda x: jnp.sqrt(jnp.abs(x)),
        ]
        x = np.random.randn(10)
        if test_complex:
            x = x + 1j * np.random.randn(10)
        for f in functions:
            for sparsity in [0, x.size]:
                with self.subTest(sparsity=sparsity, f=f):
                    y = forward_laplacian(f, sparsity)(x)
                    self.assertEqual(y.x.shape, x.shape, msg=f'{f}')
                    self.assert_allclose(y.x, f(x))
                    self.assert_allclose(y.jacobian.dense_array, self.jacobian(f, x).T)
                    self.assert_allclose(y.laplacian, self.laplacian(f, x))

    @parameterized.expand([(False,), (True,)])
    def test_binary(self, test_complex: bool):
        functions = [jnp.add, jnp.subtract, jnp.multiply, jnp.divide]
        x1 = np.random.randn(10)
        x2 = np.random.randn(10)
        if test_complex:
            x1 = x1 + 1j * np.random.randn(10)
            x2 = x2 + 1j * np.random.randn(10)
        for f in functions:
            x = jnp.stack([x1, x2])

            def wrapped_f(x):
                return f(x[0], x[1])

            def f_left(x):
                return f(x, x2)

            def f_right(x):
                return f(x2, x)

            for sparsity in [0, x1.size]:
                # test both arguments
                with self.subTest(sparsity=sparsity, f=f, binary=True):
                    y = forward_laplacian(wrapped_f, sparsity)(x)
                    self.assertEqual(y.x.shape, x1.shape, msg=f'{f}')
                    self.assert_allclose(y.x, wrapped_f(x))
                    self.assert_allclose(
                        y.jacobian.dense_array, self.jacobian(wrapped_f, x).T
                    )
                    self.assert_allclose(y.laplacian, self.laplacian(wrapped_f, x))

                # test left hand argument
                with self.subTest(sparsity=sparsity, f=f, binary=False):
                    y = forward_laplacian(f_left, sparsity)(x1)
                    self.assertEqual(y.x.shape, x1.shape, msg=f'{f}')
                    self.assert_allclose(y.x, f_left(x1))
                    self.assert_allclose(
                        y.jacobian.dense_array, self.jacobian(f_left, x1).T
                    )
                    self.assert_allclose(y.laplacian, self.laplacian(f_left, x1))

                # test right hand argument
                with self.subTest(sparsity=sparsity, f=f, binary=False):
                    y = forward_laplacian(f_right, sparsity)(x1)
                    self.assertEqual(y.x.shape, x1.shape, msg=f'{f}')
                    self.assert_allclose(y.x, f_right(x1))
                    self.assert_allclose(
                        y.jacobian.dense_array, self.jacobian(f_right, x1).T
                    )
                    self.assert_allclose(y.laplacian, self.laplacian(f_right, x1))

    @parameterized.expand([(False, False), (False, True), (True, False), (True, True)])
    def test_matmul(self, left_complex: bool, right_complex: bool):
        x = np.random.normal(size=(16,))
        w = np.random.normal(size=(16, 16))

        @jax.jit
        def f(x):
            return jnp.matmul(x, w)

        if left_complex:
            x = x * 1j
        if right_complex:
            w = w * 1j

        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                y = forward_laplacian(f, sparsity)(x)
                self.assertEqual(y.x.shape, f(x).shape)
                self.assert_allclose(y.x, f(x))
                self.assert_allclose(y.jacobian.dense_array, self.jacobian(f, x).T)
                self.assert_allclose(y.laplacian, self.laplacian(f, x))

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
                self.assert_allclose(y.x, f((a, b)))
                jac = self.jacobian(f, (a, b))
                jac = jnp.concatenate(jtu.tree_leaves(jac), axis=0)
                self.assert_allclose(y.jacobian.dense_array, jac)
                self.assert_allclose(y.laplacian, self.laplacian(f, (a, b)))

        # Test some polynomial stuff
        x = np.random.normal(size=(4,))
        w1 = np.random.normal(size=(4, 8))
        w2 = np.random.normal(size=(4, 4, 8))

        @jax.jit
        def f(x):
            return jnp.dot(x, w1) + jnp.einsum('...i,...j,...ijk->...k', x, x, w2)

        for sparsity in [0, a.size + b.size]:
            with self.subTest(sparsity=sparsity):
                y = jax.jit(forward_laplacian(f, sparsity))(x)
                self.assertEqual(y.x.shape, f(x).shape)
                self.assert_allclose(y.x, f(x))
                jac = self.jacobian(f, x)
                jac = jnp.concatenate(jtu.tree_leaves(jac), axis=0)
                self.assert_allclose(y.jacobian.dense_array.T, jac)
                self.assert_allclose(y.laplacian, self.laplacian(f, x))

    @parameterized.expand([(False,), (True,)])
    def test_slogdet(self, test_complex: bool):
        x = np.random.normal(size=(16 * 16))
        w = np.random.normal(size=(16 * 16, 16 * 16))
        if test_complex:
            w = w + 1j * np.random.normal(size=w.shape)

        @jax.jit
        def f(x):
            return jnp.linalg.slogdet(jnp.tanh((x @ w).reshape(16, 16)))

        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                sign_y, log_y = jax.jit(forward_laplacian(f, sparsity))(x)
                self.assertEqual(log_y.x.shape, f(x)[1].shape)
                self.assert_allclose(log_y.x, f(x)[1])
                self.assert_allclose(
                    log_y.jacobian.dense_array, self.jacobian(f, x)[1].T
                )
                self.assert_allclose(log_y.laplacian, self.laplacian(f, x)[1])

                self.assertEqual(sign_y.shape, log_y.x.shape)
                if test_complex:
                    self.assertIsInstance(sign_y, FwdLaplArray)
                    self.assert_allclose(
                        sign_y.jacobian.dense_array, self.jacobian(f, x)[0].T
                    )
                    self.assert_allclose(sign_y.laplacian, self.laplacian(f, x)[0])
                else:
                    self.assertIsInstance(sign_y, jax.Array)

    def test_custom_hessian(self):
        x = np.random.normal(size=(16,))

        @jax.jit
        def identity(x):
            return 1.0 * x

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
                self.assert_allclose(y.x, f(x))
                self.assert_allclose(y.jacobian.dense_array, self.jacobian(f, x).T)
                self.assert_allclose(y.laplacian, 10)
        deregister_function('identity')

    def test_dtype(self):
        x = np.random.normal(size=(16,))

        def f(x, dtype):
            return jax.lax.convert_element_type(x, dtype)

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

    def test_split(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (16,))

        def f(x):
            return jnp.split(x, 2)

        # Check that the output is still sparse
        y_fwd = forward_laplacian(f, sparsity_threshold=1)(x)
        assert y_fwd[0].jacobian.data.shape == (1, 8)
        assert y_fwd[1].jacobian.data.shape == (1, 8)
