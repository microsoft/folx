from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import folx.jvp
import folx.utils
from folx import forward_laplacian
from folx.utils import materialize_by_gather


def scatter_reference(x, idx, max_idx: int):
    """Runtime scatter equivalent of the static gather table."""

    @jax.vmap
    def aggregate(x, indices):
        return jax.ops.segment_sum(x, indices, max_idx)

    return aggregate(x, jnp.asarray(idx))


@pytest.mark.parametrize('rest', [(), (1,), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize(
    'p, k, max_idx', [(1, 1, 1), (3, 5, 3), (8, 16, 4), (64, 6, 8), (32, 24, 24)]
)
def test_gather_matches_scatter(p: int, k: int, max_idx: int, rest: tuple[int, ...]):
    # The static gather table pads with -1 and clips the indices; it must agree
    # with the scatter it replaces, including for dropped and duplicate rows.
    rng = np.random.RandomState(p * k + max_idx)
    x = rng.normal(size=(p, k, *rest)).astype(np.float32)
    idx = rng.randint(-1, max_idx, size=(p, k))
    gathered = materialize_by_gather(jnp.asarray(x), idx, max_idx)
    if gathered is None:
        pytest.skip('padded table too large, scatter is used anyway')
    np.testing.assert_allclose(
        gathered, scatter_reference(jnp.asarray(x), idx, max_idx), atol=1e-5
    )


@pytest.mark.parametrize('batched', [False, True], ids=['plain', 'vmapped'])
def test_gather_matches_scatter_all_dropped(batched: bool):
    # An all -1 mask has no valid rows and must materialize zeros.
    x = np.random.RandomState(0).normal(size=(4, 3, 2)).astype(np.float32)
    idx = np.full((4, 3), -1)
    fn = partial(materialize_by_gather, idx=idx, max_idx=5)
    x_in = jnp.asarray(x[None] if batched else x)
    gathered = jax.jit(jax.vmap(fn) if batched else fn)(x_in)
    np.testing.assert_array_equal(gathered, np.zeros((*x_in.shape[:-2], 5, 2)))


@pytest.mark.parametrize('sparsity', [0, 4, 12])
@pytest.mark.parametrize(
    'fn',
    [
        lambda x: jnp.tanh(x).sum(),
        lambda x: (jnp.tanh(x).sum(0) ** 2).sum(),
        lambda x: jnp.prod(jnp.tanh(x), axis=0).sum(),
        lambda x: (
            jnp.zeros((4, 3)).at[np.array([2, 0, 3, 1])].add(jnp.tanh(x)) ** 2
        ).sum(),
        lambda x: jnp.tanh(jnp.sqrt(((x[:, None] - x[None]) ** 2).sum(-1) + 1)).sum(),
    ],
    ids=['sum', 'sum_axis', 'prod', 'scatter', 'pairwise'],
)
def test_forward_laplacian_gather_matches_scatter(fn, sparsity: int, monkeypatch):
    # End to end: disabling the static gather table must not change results.
    x = np.random.RandomState(0).normal(size=(2, 4, 3))
    run = jax.jit(jax.vmap(forward_laplacian(fn, sparsity)))
    with_gather = run(x)
    monkeypatch.setattr(folx.utils, 'materialize_by_gather', lambda *a, **k: None)
    monkeypatch.setattr(folx.jvp, 'materialize_by_gather', lambda *a, **k: None)
    without_gather = jax.jit(jax.vmap(forward_laplacian(fn, sparsity)))(x)

    np.testing.assert_allclose(with_gather.x, without_gather.x, atol=1e-6)
    np.testing.assert_allclose(
        with_gather.jacobian.dense_array, without_gather.jacobian.dense_array, atol=1e-6
    )
    np.testing.assert_allclose(
        with_gather.laplacian, without_gather.laplacian, atol=1e-5
    )
