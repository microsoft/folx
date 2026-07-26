import functools
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from laplacian_testcase import LaplacianTestCase
from packaging.version import Version
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
                with self.subTest(sparsity=sparsity, f=getattr(f, '__name__', str(f))):
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
                with self.subTest(
                    sparsity=sparsity, f=getattr(f, '__name__', str(f)), binary=True
                ):
                    y = forward_laplacian(wrapped_f, sparsity)(x)
                    self.assertEqual(y.x.shape, x1.shape, msg=f'{f}')
                    self.assert_allclose(y.x, wrapped_f(x))
                    self.assert_allclose(
                        y.jacobian.dense_array, self.jacobian(wrapped_f, x).T
                    )
                    self.assert_allclose(y.laplacian, self.laplacian(wrapped_f, x))

                # test left hand argument
                with self.subTest(
                    sparsity=sparsity, f=getattr(f, '__name__', str(f)), binary=False
                ):
                    y = forward_laplacian(f_left, sparsity)(x1)
                    self.assertEqual(y.x.shape, x1.shape, msg=f'{f}')
                    self.assert_allclose(y.x, f_left(x1))
                    self.assert_allclose(
                        y.jacobian.dense_array, self.jacobian(f_left, x1).T
                    )
                    self.assert_allclose(y.laplacian, self.laplacian(f_left, x1))

                # test right hand argument
                with self.subTest(
                    sparsity=sparsity, f=getattr(f, '__name__', str(f)), binary=False
                ):
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

    def test_matmul_both_sparse(self):
        # Regression test for #35: matmul where both operands have sparse
        # Jacobians and the broadcast intermediate is larger than the inputs.
        n, d = 8, 3
        x = np.random.normal(size=(n, d))

        @jax.jit
        def attn(x):
            s = jnp.exp(jnp.matmul(x, jnp.swapaxes(x, -2, -1)))
            return jnp.matmul(s, x)

        @jax.jit
        def rect(x):
            u = jnp.sin(x)  # (n, d)
            v = jnp.cos(jnp.swapaxes(x, -2, -1))  # (d, n)
            return jnp.matmul(u, v)

        for f in [attn, rect]:
            for sparsity in [0, x.size]:
                with self.subTest(f=f.__name__, sparsity=sparsity):
                    y = forward_laplacian(f, sparsity)(x)
                    out_shape = f(x).shape
                    self.assertEqual(y.x.shape, out_shape)
                    self.assert_allclose(y.x, f(x))
                    jac = self.jacobian(f, x).reshape(*out_shape, x.size)
                    self.assert_allclose(
                        y.jacobian.dense_array, np.moveaxis(jac, -1, 0)
                    )
                    self.assert_allclose(
                        y.laplacian, self.laplacian(f, x).reshape(out_shape)
                    )

    def test_matmul_degenerate_broadcast_shapes(self):
        # einsum patterns whose dot_general has a size-1 or rank-mismatched
        # broadcast dim; the mul-sum decomposition must keep the output shape.
        k = 3
        x = np.random.normal(size=(5, k))

        cases = [
            lambda x: jnp.einsum('nk,mk->nm', x, jnp.sin(x[:1])),  # (5, 1)
            lambda x: jnp.einsum('nk,mk->nm', x[:1], jnp.sin(x)),  # (1, 5)
            lambda x: jnp.einsum('k,mk->m', x[0], jnp.sin(x[:k])),  # (k,)
            # batch-last message aggregation (FiRE/MPNN pattern)
            lambda x: jnp.einsum(
                'ijh,jh->ih', jnp.sin(x[:, None, :] * x[None, :, :]), jnp.cos(x)
            ),
        ]
        for i, f in enumerate(cases):
            for sparsity in [0, x.size]:
                with self.subTest(case=i, sparsity=sparsity):
                    y = forward_laplacian(f, sparsity)(x)
                    out_shape = f(x).shape
                    self.assertEqual(y.x.shape, out_shape)
                    self.assert_allclose(y.x, f(x))
                    # dense_array may be narrower than x.size if trailing
                    # inputs are unused; pad it for comparison.
                    dense_jac = np.asarray(y.jacobian.dense_array)
                    pad = x.size - dense_jac.shape[0]
                    dense_jac = np.pad(
                        dense_jac, ((0, pad),) + ((0, 0),) * (dense_jac.ndim - 1)
                    )
                    jac = self.jacobian(f, x).reshape(*out_shape, x.size)
                    self.assert_allclose(dense_jac, np.moveaxis(jac, -1, 0))
                    self.assert_allclose(
                        y.laplacian, self.laplacian(f, x).reshape(out_shape)
                    )

    def test_matmul_preferred_element_type(self):
        # The sparse mul-sum path must honor dot_general's accumulation dtype.
        # Seeded and scaled: bf16 rounding through exp is sensitive to magnitude.
        x = (0.5 * np.random.default_rng(0).normal(size=(4, 2))).astype(jnp.bfloat16)

        def f(x):
            s = jnp.exp(jnp.matmul(x, jnp.swapaxes(x, -2, -1)))
            return jax.lax.dot_general(
                s, x, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
            )

        try:
            jax.block_until_ready(jax.jit(f)(x))
        except Exception as e:  # old jaxlib CPU backends lack bf16->f32 dots
            self.skipTest(f'bf16 dot_general with f32 accumulation unsupported: {e}')

        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                y = forward_laplacian(f, sparsity)(x)
                self.assertEqual(y.x.dtype, jnp.float32)
                self.assertEqual(y.jacobian.data.dtype, jnp.float32)
                self.assertEqual(y.laplacian.dtype, jnp.float32)
                ref = forward_laplacian(f, 0)(x.astype(jnp.float32))
                self.assert_allclose(y.x, ref.x, rtol=5e-2)
                self.assert_allclose(y.laplacian, ref.laplacian, rtol=5e-2)

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
        def _f(w, x):
            return jnp.linalg.slogdet(jnp.tanh((x @ w).reshape(16, 16)))

        f = partial(_f, w)

        for sparsity in [0, x.size]:
            for use_shard_map in [False, True]:
                with self.subTest(sparsity=sparsity, use_shard_map=use_shard_map):
                    if use_shard_map and (
                        Version(jax.__version__) < Version('0.7.1')
                        or (
                            sparsity != 0
                            and Version(jax.__version__) < Version('0.7.2')
                        )
                    ):
                        self.skipTest('jax version too old')
                    if use_shard_map:
                        mesh = jax.sharding.Mesh(
                            jax.devices()[:1],
                            'i',
                            axis_types=jax.sharding.AxisType.Explicit,
                        )

                        @jax.jit
                        @partial(
                            jax.shard_map,
                            in_specs=(jax.P(), jax.P('i')),
                            out_specs=jax.P('i'),
                        )
                        @partial(jax.vmap, in_axes=(None, 0))
                        def forward_laplacian_sh(w, x):
                            return forward_laplacian(partial(_f, w), sparsity)(x)

                        with jax.set_mesh(mesh):
                            x_sh = jax.sharding.reshard(x[None], jax.P('i'))
                            w_sh = jax.sharding.reshard(w, jax.P())
                            sign_y, log_y = jax.tree.map(
                                lambda x: x[0], forward_laplacian_sh(w_sh, x_sh)
                            )
                    else:
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
                    del sign_y
                    del log_y

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
            with self.subTest(dtype=dtype.__name__):
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
            with self.subTest(dtype=dtype.__name__):
                y = jax.jit(forward_laplacian(functools.partial(f, dtype=dtype)))(x)
                self.assertIsInstance(y, jax.Array)

    def test_multi_output(self):
        # Regression test for #29: multi-output functions produce pytree
        # gradients, so a single input mask must be broadcast to every output.
        x = np.random.randn(8)

        @jax.custom_jvp
        def sincos(x):
            return jnp.sin(x), jnp.cos(x)

        @sincos.defjvp
        def sincos_jvp(primals, tangents):
            ((x,), (dx,)) = primals, tangents
            return sincos(x), (jnp.cos(x) * dx, -jnp.sin(x) * dx)

        @jax.custom_jvp
        def sumdiff(a, b):
            return a + b, a - b

        @sumdiff.defjvp
        def sumdiff_jvp(primals, tangents):
            ((a, b), (da, db)) = primals, tangents
            return sumdiff(a, b), (da + db, da - db)

        @jax.jit
        def sincos_elemwise(x):
            return jnp.sin(x), jnp.cos(x)

        @jax.jit
        def sumdiff_elemwise(a, b):
            return a + b, a - b

        register_function(
            'sincos_elemwise',
            wrap_forward_laplacian(
                lambda x: (jnp.sin(x), jnp.cos(x)), in_axes=(), name='sincos_elemwise'
            ),
        )
        register_function(
            'sumdiff_elemwise',
            wrap_forward_laplacian(
                lambda a, b: (a + b, a - b), in_axes=(), name='sumdiff_elemwise'
            ),
        )

        def single_arg(x):
            return sincos(x)

        def multi_arg(x):
            return sumdiff(jnp.sin(x), jnp.cos(x))

        def single_arg_elemwise(x):
            return sincos_elemwise(x)

        def multi_arg_elemwise(x):
            return sumdiff_elemwise(jnp.sin(x), jnp.cos(x))

        try:
            fns = [single_arg, multi_arg, single_arg_elemwise, multi_arg_elemwise]
            for f in fns:
                for sparsity in [0, x.size]:
                    with self.subTest(f=f.__name__, sparsity=sparsity):
                        y = forward_laplacian(f, sparsity)(x)
                        self.assertEqual(len(y), 2)
                        for i, y_i in enumerate(y):

                            def f_i(x, f=f, i=i):
                                return f(x)[i]

                            self.assert_allclose(y_i.x, f_i(x))
                            self.assert_allclose(
                                y_i.jacobian.dense_array, self.jacobian(f_i, x).T
                            )
                            self.assert_allclose(y_i.laplacian, self.laplacian(f_i, x))
        finally:
            deregister_function('sincos_elemwise')
            deregister_function('sumdiff_elemwise')

    def test_split(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (16,))

        def f(x):
            return jnp.split(x, 2)

        # Check that the output is still sparse
        y_fwd = forward_laplacian(f, sparsity_threshold=1)(x)
        assert y_fwd[0].jacobian.data.shape == (1, 8)
        assert y_fwd[1].jacobian.data.shape == (1, 8)

    def check_forward_laplacian(self, f, x, sparsity):
        def flat_f(z):
            return f(z).reshape(-1)

        y = forward_laplacian(f, sparsity)(x)
        self.assert_allclose(y.x, f(x))
        jac = np.moveaxis(np.asarray(y.jacobian.dense_array), 0, -1)
        jac = jac.reshape(y.x.size, -1)
        expected = self.jacobian(flat_f, x)
        # The dense jacobian is truncated at the highest referenced input.
        self.assert_allclose(jac, expected[:, : jac.shape[1]])
        self.assert_allclose(expected[:, jac.shape[1] :], 0)
        self.assert_allclose(np.asarray(y.laplacian).reshape(-1), self.laplacian(f, x))
        return y

    def test_pad(self):
        # https://github.com/microsoft/folx/issues/27
        x = np.random.randn(6)
        cases = [
            lambda z: jnp.pad(z.reshape(2, 3), ((1, 0), (0, 2))),
            lambda z: jax.lax.pad(z.reshape(2, 3), 7.0, ((1, 0, 1), (0, 2, 0))),
            lambda z: jax.lax.pad(z[:5], z[5], ((2, 1, 1),)),  # varying pad value
            lambda z: jax.lax.pad(z, 0.0, ((-1, -2, 0),)),  # negative padding
        ]
        for i, f in enumerate(cases):
            for sparsity in [0, x.size]:
                with self.subTest(case=i, sparsity=sparsity):
                    y = self.check_forward_laplacian(f, x, sparsity)
                    if sparsity:
                        self.assertTrue(y.is_jacobian_weak)

    def test_add_any(self):
        # jax.grad inside the function emits the add_any primitive
        x = np.random.randn(6)

        def f(z):
            return jax.grad(lambda y: jnp.sin(y).sum() + (y**2).sum())(z) * z

        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                self.check_forward_laplacian(f, x, sparsity)

    def test_elementwise_real(self):
        # Elementwise primitives that only support real inputs
        functions = [
            jax.scipy.special.erf,
            jax.scipy.special.erfc,
            lambda z: jax.scipy.special.erfinv(jnp.tanh(z)),
            jnp.sinh,
            jnp.cosh,
            jnp.arcsinh,
            lambda z: jnp.arccosh(jnp.abs(z) + 1.5),
            lambda z: jnp.arctanh(jnp.tanh(z)),
            jnp.cbrt,
            jnp.exp2,
            lambda z: jax.scipy.special.gammaln(jnp.abs(z) + 0.5),
            lambda z: jax.scipy.special.digamma(jnp.abs(z) + 0.5),
            lambda z: jax.nn.gelu(z, approximate=False),
        ]
        x = np.random.randn(6)
        for f in functions:
            for sparsity in [0, x.size]:
                with self.subTest(f=getattr(f, '__name__', str(f)), sparsity=sparsity):
                    self.check_forward_laplacian(f, x, sparsity)

    def test_piecewise(self):
        # Piecewise linear/constant primitives with zero Hessian a.e.
        functions = [
            lambda z: jnp.clip(z, -0.5, 0.5),
            lambda z: jax.lax.rem(z, 0.7),
            lambda z: jnp.floor(z * 3) * z,
            lambda z: jnp.ceil(z * 3) * z,
            lambda z: jnp.round(z * 3) * z,
            lambda z: jax.lax.cummax(z, axis=0),
            lambda z: jax.lax.cummin(z, axis=0),
        ]
        x = np.random.randn(6)
        for i, f in enumerate(functions):
            for sparsity in [0, x.size]:
                with self.subTest(case=i, sparsity=sparsity):
                    self.check_forward_laplacian(f, x, sparsity)

    def test_cumprod(self):
        # cumprod is shape-preserving but not elementwise; its Hessian has
        # off-diagonal blocks and must not take the elementwise JHJ fast path.
        x = np.random.randn(6)

        def f(z):
            return jnp.cumprod(jnp.sin(z))

        for sparsity in [0, x.size]:
            with self.subTest(sparsity=sparsity):
                self.check_forward_laplacian(f, x, sparsity)

    def test_indexing_primitives(self):
        functions = [
            lambda z: jnp.stack([z, z * 2], axis=1),
            lambda z: jnp.stack([z, jnp.ones(6)], axis=0),
            lambda z: jnp.unstack(z.reshape(2, 3))[1],
            lambda z: jnp.tile(z.reshape(2, 3), (2, 2)),
            lambda z: jnp.copy(z) * z,
            lambda z: jax.lax.dynamic_update_slice(z, z[:2] ** 2, (3,)),
            lambda z: jax.lax.dynamic_update_slice(jnp.zeros(8), z**2, (3,)),
        ]
        x = np.random.randn(6)
        for i, f in enumerate(functions):
            for sparsity in [0, x.size]:
                with self.subTest(case=i, sparsity=sparsity):
                    self.check_forward_laplacian(f, x, sparsity)

    def test_indexing_with_constants_stays_sparse(self):
        # Constant operands are masked with int32 fill values; the masks must
        # match that dtype or the op silently falls back to a dense jacobian.
        functions = [
            lambda z: jnp.concatenate([z, jnp.ones(4)]),
            lambda z: jnp.stack([z, jnp.ones(8)]),
            lambda z: jnp.pad(z, (1, 2)),
        ]
        x = jax.random.normal(jax.random.PRNGKey(0), (8,))
        for i, f in enumerate(functions):
            with self.subTest(case=i):
                y = forward_laplacian(f, sparsity_threshold=1)(x)
                self.assertTrue(y.is_jacobian_weak)
                self.assertEqual(y.jacobian.data.shape[0], 1)
