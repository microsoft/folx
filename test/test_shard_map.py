from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from packaging.version import Version

from folx import forward_laplacian


@pytest.mark.skipif(
    Version(jax.__version__) < Version('0.7.1'), reason='jax version too old'
)
def test_shard_map_bug_integer_pow():
    # see https://github.com/microsoft/folx/issues/38

    def f(w, x):
        return jax.lax.integer_pow(x @ w, 1)

    @jax.smap(out_axes=0, in_axes=(None, 0), axis_name='i')
    @partial(jax.vmap, in_axes=(None, 0))
    def test(w, x):
        return forward_laplacian(partial(f, w))(x)

    x = jnp.ones((1, 16))
    w = jnp.ones((16, 16))

    with jax.set_mesh(jax.sharding.Mesh(jax.devices()[:1], 'i')):
        test(w, x)


@pytest.mark.skipif(
    Version(jax.__version__) < Version('0.7.2'), reason='jax version too old'
)
@pytest.mark.skipif(len(jax.devices()) < 2, reason='needs at least two devices')
@pytest.mark.parametrize(
    'fn',
    [
        lambda x: jnp.tanh(x).sum(),
        lambda x: jnp.linalg.slogdet(jnp.tanh(x @ x.T))[1],
        lambda x: jnp.prod(jnp.tanh(x), axis=0).sum(),
    ],
    ids=['sum', 'slogdet', 'prod'],
)
@pytest.mark.parametrize('explicit', [False, True], ids=['auto', 'explicit'])
def test_shard_map_multi_device(fn, explicit: bool):
    # The sparse index bookkeeping is evaluated eagerly at trace time, which
    # must not inherit the manual mesh of the enclosing shard_map as eager
    # evaluation is only possible for a single device.
    x = np.random.RandomState(0).normal(size=(4, 3, 2))
    axis_types = {'axis_types': jax.sharding.AxisType.Explicit} if explicit else {}
    mesh = jax.sharding.Mesh(jax.devices()[:2], 'i', **axis_types)

    @jax.jit
    @partial(jax.shard_map, in_specs=(jax.P('i'),), out_specs=jax.P('i'))
    @jax.vmap
    def fwd_lapl(x):
        return forward_laplacian(fn, x.size)(x)

    with jax.set_mesh(mesh):
        y = fwd_lapl(jax.device_put(x, jax.NamedSharding(mesh, jax.P('i'))))

    lapl = jax.vmap(lambda x: jnp.trace(jax.hessian(fn)(x).reshape(x.size, x.size)))(x)
    np.testing.assert_allclose(y.x, jax.vmap(fn)(x), atol=1e-6)
    np.testing.assert_allclose(y.laplacian, lapl, atol=1e-6)
