"""Scaling of the Laplacian of a deep set over electrons (Figure 2 of the paper).

The same MLP phi maps the coordinates of every electron to a feature vector, the features
are summed over electrons, and a second MLP rho maps the sum to a scalar:
f(r) = rho(sum_i phi(r_i)). Each per-electron feature depends on only three inputs.

Usage: python resources/scripts/deepset.py [--x64] [--n-electrons 4 8 ...]
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from common import run_scaling, warmup

N_LAYERS_PHI = 4
N_LAYERS_RHO = 2
WIDTH = 256
BATCH = 512
SPARSITY_THRESHOLD = 3
# Chunk sizes of folx.batched_vmap per number of electrons, chosen to fit the 16 GB GPU.
CHUNK_SIZES = {4: 512, 8: 256, 16: 128, 32: 64, 64: 16, 128: 8}


def mlp_init(key, dims):
    keys = jax.random.split(key, len(dims) - 1)
    return [
        (jax.random.normal(k, (d_in, d_out)) / np.sqrt(d_in), jnp.zeros(d_out))
        for k, d_in, d_out in zip(keys, dims[:-1], dims[1:])
    ]


def mlp(params, x):
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


def make_deep_set(n_electrons, key):
    k_phi, k_rho = jax.random.split(key)
    phi = mlp_init(k_phi, [3] + [WIDTH] * N_LAYERS_PHI)
    rho = mlp_init(k_rho, [WIDTH] * N_LAYERS_RHO + [1])

    def f(r):
        r = r.reshape(n_electrons, 3)
        h = jnp.tanh(mlp(phi, r))
        return mlp(rho, h.sum(0))[0]

    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x64', action='store_true')
    parser.add_argument(
        '--n-electrons', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['hessian_loop', 'folx_dense_chunked', 'folx_sparse_chunked'],
    )
    args = parser.parse_args()
    if args.x64:
        jax.config.update('jax_enable_x64', True)
    warmup()
    run_scaling(
        'deepset',
        make_deep_set,
        args.n_electrons,
        args.methods,
        SPARSITY_THRESHOLD,
        CHUNK_SIZES,
        BATCH,
        args.x64,
        dict(n_layers_phi=N_LAYERS_PHI, n_layers_rho=N_LAYERS_RHO, width=WIDTH),
    )


if __name__ == '__main__':
    main()
