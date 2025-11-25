from functools import partial

import jax
import jax.numpy as jnp

from folx import forward_laplacian


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
