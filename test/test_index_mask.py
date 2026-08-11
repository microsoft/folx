"""Tests for the sparse-to-dense index mapping of FwdJacobian."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from folx.api import FwdJacobian
from folx.utils import static_index_mask

jax.config.update('jax_enable_x64', True)


def _brute_force(mask: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """First matching output row per Jacobian row, by explicit comparison."""
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


def test_static_index_mask_chunked():
    rng = np.random.default_rng(0)
    mask = rng.integers(-1, 20, size=(4, 300))
    outputs = rng.integers(-1, 20, size=(9, 300))
    np.testing.assert_array_equal(
        static_index_mask(mask, outputs, chunk=64), _brute_force(mask, outputs)
    )


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
