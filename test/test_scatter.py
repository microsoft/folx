import jax
import jax.numpy as jnp
import numpy as np
from laplacian_testcase import LaplacianTestCase

from folx import forward_laplacian


class TestScatter(LaplacianTestCase):
    def full_jacobian(self, y, n):
        """Materializes the Jacobian of a FwdLaplArray padded to n input rows."""
        jac = y.jacobian.dense_array
        if jac.shape[0] < n:
            pad = jnp.zeros((n - jac.shape[0], *jac.shape[1:]), dtype=jac.dtype)
            jac = jnp.concatenate([jac, pad], axis=0)
        return jnp.moveaxis(jac, 0, -1)

    def check_cases(self, x, cases, expect_weak=True):
        for name, f in cases:
            for sparsity in [0, x.size]:
                with self.subTest(f=name, sparsity=sparsity):
                    y = forward_laplacian(f, sparsity)(x)
                    self.assert_allclose(y.x, f(x))
                    self.assert_allclose(
                        self.full_jacobian(y, x.size), self.jacobian(f, x)
                    )
                    self.assert_allclose(
                        y.laplacian, self.laplacian(f, x).reshape(y.x.shape)
                    )
                    if expect_weak and sparsity == x.size:
                        self.assertIsNotNone(y.jacobian.x0_idx)

    def test_scatter_set(self):
        x = np.random.randn(12)
        idx = jnp.array([0, 2, 5])

        def rows_into_zeros(x):
            # issue #28 pattern
            r = x.reshape(-1, 3)
            d = jnp.sqrt(jnp.sum(jnp.square(r), axis=1))
            sph = jnp.zeros((2, d.shape[0]))
            sph = sph.at[0].set(d)
            sph = sph.at[1].set(d * 5.0)
            return sph

        cases = [
            ('rows_into_zeros', rows_into_zeros),
            ('fancy_set', lambda x: jnp.sin(x).at[idx].set(x[:3] * 2)),
            ('const_updates', lambda x: jnp.sin(x).at[idx].set(1.5)),
            ('scalar_index', lambda x: jnp.sin(x).at[3].set(x[0] * x[1])),
            (
                'oob_dropped',
                lambda x: jnp.sin(x).at[jnp.array([1, 100])].add(x[:2] * 3),
            ),
        ]
        self.check_cases(x, cases)

    def test_scatter_add(self):
        x = np.random.randn(12)
        dup_idx = jnp.array([1, 1, 3])

        def segment_sum(x):
            r = x.reshape(-1, 3)
            d = jnp.sqrt(jnp.sum(jnp.square(r), axis=1))
            return jnp.zeros((2,)).at[jnp.array([0, 1, 0, 1])].add(d)

        def windowed_add(x):
            r = jnp.sin(x).reshape(4, 3)
            return r.at[jnp.array([0, 2])].add(jnp.cos(x[:6]).reshape(2, 3))

        def multi_dim_indices(x):
            r = jnp.sin(x).reshape(4, 3)
            i, j = jnp.array([0, 3, 0]), jnp.array([1, 2, 1])
            return r.at[i, j].add(x[:3] * x[3:6])

        cases = [
            ('segment_sum', segment_sum),
            ('duplicate_add', lambda x: jnp.sin(x).at[dup_idx].add(x[:3] ** 2)),
            ('windowed_add', windowed_add),
            ('multi_dim_indices', multi_dim_indices),
            ('const_updates', lambda x: jnp.sin(x).at[dup_idx].add(2.5)),
        ]
        self.check_cases(x, cases)

    def test_scatter_min_max(self):
        x = np.random.randn(12)
        idx = jnp.array([0, 2, 5])
        cases = [
            ('min', lambda x: jnp.sin(x).at[idx].min(x[6:9] * 2)),
            ('max', lambda x: jnp.sin(x).at[idx].max(x[6:9] * 2)),
            ('min_const', lambda x: jnp.sin(x).at[idx].min(0.1)),
            ('max_const', lambda x: jnp.sin(x).at[idx].max(-0.1)),
        ]
        self.check_cases(x, cases)

    def test_scatter_mixed_sparsity(self):
        # Mixed dense/sparse Jacobians: cumsum densifies at threshold 4, so one
        # argument is dense while the other stays sparse.
        x = np.random.randn(12)
        idx = jnp.array([0, 2, 5])
        cases = [
            ('dense_op_set', lambda x: jnp.cumsum(x).at[idx].set(x[:3] * 2)),
            ('dense_upd_set', lambda x: jnp.sin(x).at[idx].set(jnp.cumsum(x)[:3])),
            ('dense_op_add', lambda x: jnp.cumsum(x).at[idx].add(x[:3] * 2)),
            ('dense_upd_add', lambda x: jnp.sin(x).at[idx].add(jnp.cumsum(x)[:3])),
            ('dense_upd_min', lambda x: jnp.sin(x).at[idx].min(jnp.cumsum(x)[:3])),
        ]
        for name, f in cases:
            for sparsity in [0, 4, x.size]:
                with self.subTest(f=name, sparsity=sparsity):
                    y = forward_laplacian(f, sparsity)(x)
                    self.assert_allclose(y.x, f(x))
                    self.assert_allclose(
                        self.full_jacobian(y, x.size), self.jacobian(f, x)
                    )
                    self.assert_allclose(
                        y.laplacian, self.laplacian(f, x).reshape(y.x.shape)
                    )

    def test_dynamic_update_slice(self):
        x = np.random.randn(12)
        cases = [
            (
                'dus',
                lambda x: jax.lax.dynamic_update_slice(jnp.sin(x), x[:3] * 2, (4,)),
            ),
            (
                'dus_const',
                lambda x: jax.lax.dynamic_update_slice(jnp.sin(x), jnp.ones(3), (4,)),
            ),
        ]
        self.check_cases(x, cases)

    def test_sparsity_breaking_scatter_add(self):
        # A per-nucleus segment sum makes every output depend on every input, so
        # the scatter output is densified. Tracing this must not scale with
        # (positions x rows x dense rows).
        n_el, n_nuc = 8, 3
        el = np.repeat(np.arange(n_el), n_nuc)
        nuc = np.tile(np.arange(n_nuc), n_el)

        def f(x):
            r = x.reshape(n_el, 3)
            d = jnp.linalg.norm(r[el] - jnp.arange(n_nuc)[nuc, None], axis=-1)
            return jnp.zeros((n_nuc,)).at[nuc].add(jnp.tanh(d))

        x = np.random.randn(n_el * 3)
        self.check_cases(x, [('nuc_segment_sum', f)], expect_weak=False)

    def test_scatter_dense_materialization(self):
        # Densifying a scatter output must agree with the dense-from-the-start
        # path for set/add/min alike.
        x = np.random.randn(12)
        idx = jnp.array([0, 1, 0, 1])

        def build(op):
            def f(x):
                r = jnp.sin(x).reshape(4, 3)
                base = jnp.zeros((2, 3))
                return getattr(base.at[idx], op)(r)

            return f

        for op in ['add', 'set', 'min', 'max']:
            with self.subTest(op=op):
                f = build(op)
                dense = forward_laplacian(f, 0)(x)
                sparse = forward_laplacian(f, 2)(x)
                self.assert_allclose(sparse.x, dense.x)
                self.assert_allclose(sparse.laplacian, dense.laplacian)
                self.assert_allclose(
                    self.full_jacobian(sparse, x.size),
                    self.full_jacobian(dense, x.size),
                )

    def test_scatter_jit_laplacian(self):
        # End-to-end check of issue #28 under jit: the scatter must retain
        # sparsity and match the dense reference.
        def fwd(x):
            r = x.reshape(-1, 3)
            d = jnp.sqrt(jnp.sum(jnp.square(r), axis=1))
            sph = jnp.zeros((2, d.shape[0]))
            sph = sph.at[0].set(d)
            sph = sph.at[1].set(d * 5.0)
            return jnp.sum(sph)

        x = np.asarray(jax.random.normal(jax.random.PRNGKey(12), (30,)))
        y_sparse = jax.jit(forward_laplacian(fwd, 6))(x)
        y_dense = jax.jit(forward_laplacian(fwd, 0))(x)
        self.assert_allclose(y_sparse.x, y_dense.x)
        self.assert_allclose(y_sparse.laplacian, y_dense.laplacian)
        self.assert_allclose(y_sparse.laplacian, self.laplacian(fwd, x))
