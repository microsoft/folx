"""Tests for the sparse-to-dense index mapping of FwdJacobian."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from folx.api import FwdJacobian
from folx.utils import compact_repeated_dims_except, static_index_mask

jax.config.update('jax_enable_x64', True)


def _brute_force(mask: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """First matching output row per Jacobian row, by explicit comparison."""
    shape = np.broadcast_shapes(mask.shape[1:], outputs.shape[1:])

    def expand(a):
        pad = (1,) * (len(shape) - (a.ndim - 1))
        return np.broadcast_to(
            a.reshape(a.shape[0], *pad, *a.shape[1:]), (a.shape[0], *shape)
        )

    mask, outputs = expand(mask), expand(outputs)
    matching = mask[:, None] == outputs[None, :]
    return np.where(matching.any(1), matching.argmax(1), -1)


@pytest.mark.parametrize('shape', [(5,), (3, 4), (2, 3, 4)])
@pytest.mark.parametrize('k,n', [(1, 1), (2, 3), (5, 8), (4, 2)])
@pytest.mark.parametrize('low', [-2, -1, 0])
def test_static_index_mask(shape, k, n, low):
    rng = np.random.default_rng(k * 100 + n * 10 - low)
    mask = rng.integers(low, 6, size=(k, *shape))
    outputs = rng.integers(low, 6, size=(n, *shape))
    np.testing.assert_array_equal(
        static_index_mask(mask, outputs), _brute_force(mask, outputs)
    )


@pytest.mark.parametrize(
    'mask_shape,out_shape',
    [
        ((4, 5), (1, 5)),  # outputs constant along a leading axis
        ((4, 5), (4, 1)),  # outputs constant along a trailing axis
        ((3, 4, 5), (1, 1, 5)),
        ((3, 4, 5), (5,)),  # fewer axes, right aligned
    ],
)
def test_static_index_mask_broadcast(mask_shape, out_shape):
    """Outputs may be a broadcastable prefix of the mask's position frame."""
    rng = np.random.default_rng(len(out_shape))
    mask = rng.integers(-1, 6, size=(3, *mask_shape))
    outputs = rng.integers(-1, 6, size=(4, *out_shape))
    np.testing.assert_array_equal(
        static_index_mask(mask, outputs), _brute_force(mask, outputs)
    )


def test_static_index_mask_chunked():
    rng = np.random.default_rng(0)
    mask = rng.integers(-1, 20, size=(4, 300))
    outputs = rng.integers(-1, 20, size=(9, 300))
    np.testing.assert_array_equal(
        static_index_mask(mask, outputs, chunk=64), _brute_force(mask, outputs)
    )


@pytest.mark.parametrize('seed', range(6))
def test_compaction_preserves_unique(seed):
    """Dropping constant axes must not change which mask rows are unique.

    sparse_diag_jvp compacts before ``np.unique(axis=0)``; both the grouping and
    the row order have to survive it.
    """
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 6))
    shape = tuple(int(s) for s in rng.integers(1, 4, size=int(rng.integers(1, 4))))
    a = rng.integers(-1, 4, size=(k, *shape))
    for d in range(1, a.ndim):
        if rng.random() < 0.5:
            a = np.repeat(np.take(a, [0], axis=d), a.shape[d], axis=d)

    u_full, inv_full = np.unique(a, axis=0, return_inverse=True)
    u_c, inv_c = np.unique(
        compact_repeated_dims_except(a, axis=0)[0], axis=0, return_inverse=True
    )
    np.testing.assert_array_equal(inv_full.reshape(-1), inv_c.reshape(-1))
    np.testing.assert_array_equal(np.broadcast_to(u_c, u_full.shape), u_full)


@pytest.mark.parametrize('shape', [(4,), (3, 2)])
@pytest.mark.parametrize('k', [1, 3])
def test_dense_array(shape, k):
    """Densifying must place each row at the index its mask names."""
    rng = np.random.default_rng(k)
    mask = rng.integers(-1, 5, size=(k, *shape))
    data = rng.normal(size=(k, *shape))
    dense = np.asarray(FwdJacobian(jnp.asarray(data), mask).dense_array)

    expected = np.zeros((mask.max() + 1, *shape))
    for i in np.ndindex(*mask.shape):
        if mask[i] >= 0:
            expected[(mask[i], *i[1:])] += data[i]
    np.testing.assert_allclose(dense, expected, rtol=1e-12, atol=1e-12)


def test_dense_array_all_invalid():
    """A Jacobian without any dependency densifies to a single zero row."""
    mask = np.full((2, 3), -1)
    dense = FwdJacobian(jnp.ones((2, 3)), mask).dense_array
    assert dense.shape == (1, 3)
    np.testing.assert_allclose(dense, 0)
