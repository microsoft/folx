"""Scaling of the Laplacian of a FermiNet-style wave function (Figure 2 of the paper).

The wave function follows FermiNet: one-electron streams h_i and two-electron streams
h_ij interact via permutation-equivariant mean aggregation over 4 layers, followed by
spin-block orbitals with exponential envelopes and a single determinant per spin.
We compute the Laplacian of log|psi| for N electrons in a chain of N hydrogen atoms.

Usage: python resources/scripts/ferminet.py [--x64] [--n-electrons 4 8 ...]
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from common import run_scaling, warmup

N_LAYERS = 4
WIDTH_ONE = 256
WIDTH_TWO = 32
BATCH = 512
SPARSITY_THRESHOLD = 6
# Batch chunk size of the chunked variant, see folx.batched_vmap.
# Chunk sizes of folx.batched_vmap per number of electrons, chosen to fit the 16 GB GPU.
CHUNK_SIZES = {4: 512, 8: 256, 16: 128, 32: 64, 64: 16, 128: 8}


def dense_init(key, d_in, d_out):
    w = jax.random.normal(key, (d_in, d_out)) / np.sqrt(d_in)
    return w, jnp.zeros(d_out)


def init_params(key, n_atoms):
    keys = iter(jax.random.split(key, 4 * N_LAYERS + 4))
    d_one, d_two = 4 * n_atoms, 4
    layers = []
    for _ in range(N_LAYERS):
        layers.append(
            dict(
                one=dense_init(next(keys), 3 * d_one + 2 * d_two, WIDTH_ONE),
                two=dense_init(next(keys), d_two, WIDTH_TWO),
            )
        )
        d_one, d_two = WIDTH_ONE, WIDTH_TWO
    return dict(layers=layers, d_last=d_one, keys=next(keys))


def make_wave_function(n_electrons, key):
    n_up = n_electrons // 2
    atoms = jnp.stack(
        [jnp.arange(n_electrons) * 1.4, jnp.zeros(n_electrons), jnp.zeros(n_electrons)],
        axis=-1,
    )
    params = init_params(key, n_atoms=n_electrons)
    k_orb, k_env = jax.random.split(params['keys'])
    orbital_w = jax.random.normal(k_orb, (params['d_last'], n_electrons)) / np.sqrt(
        params['d_last']
    )
    env_pi = jax.random.normal(k_env, (n_electrons, n_electrons)) / np.sqrt(n_electrons)
    env_sigma = jnp.ones((n_electrons, n_electrons))

    def spin_means(h):
        return h[:n_up].mean(0), h[n_up:].mean(0)

    def log_psi(r):
        r = r.reshape(n_electrons, 3)
        r_ae = r[:, None] - atoms[None]
        d_ae = jnp.linalg.norm(r_ae, axis=-1, keepdims=True)
        r_ee = r[:, None] - r[None]
        # Avoid the non-differentiable norm on the diagonal.
        d_ee = jnp.sqrt(
            (r_ee**2).sum(-1, keepdims=True) + jnp.eye(n_electrons)[..., None]
        )
        h_one = jnp.concatenate([r_ae, d_ae], axis=-1).reshape(n_electrons, -1)
        h_two = jnp.concatenate([r_ee, d_ee], axis=-1)
        for layer in params['layers']:
            up, down = spin_means(h_one)
            two_up = h_two[:, :n_up].mean(1)
            two_down = h_two[:, n_up:].mean(1)
            features = jnp.concatenate(
                [
                    h_one,
                    jnp.broadcast_to(up, h_one.shape),
                    jnp.broadcast_to(down, h_one.shape),
                    two_up,
                    two_down,
                ],
                axis=-1,
            )
            w, b = layer['one']
            new_one = jnp.tanh(features @ w + b)
            w, b = layer['two']
            new_two = jnp.tanh(h_two @ w + b)
            h_one = new_one + h_one if new_one.shape == h_one.shape else new_one
            h_two = new_two + h_two if new_two.shape == h_two.shape else new_two
        # envelope[i, k] = sum_I pi_kI exp(-sigma_kI |r_i - R_I|)
        envelope = (env_pi * jnp.exp(-env_sigma * d_ae[:, None, :, 0])).sum(-1)
        orbitals = (h_one @ orbital_w) * envelope
        log_abs = 0.0
        for start, stop in [(0, n_up), (n_up, n_electrons)]:
            log_abs += jnp.linalg.slogdet(orbitals[start:stop, start:stop])[1]
        return log_abs

    return log_psi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x64', action='store_true')
    parser.add_argument(
        '--n-electrons', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['hessian_loop', 'folx_sparse', 'folx_sparse_chunked'],
    )
    args = parser.parse_args()
    if args.x64:
        jax.config.update('jax_enable_x64', True)
    warmup()
    run_scaling(
        'ferminet',
        make_wave_function,
        args.n_electrons,
        args.methods,
        SPARSITY_THRESHOLD,
        CHUNK_SIZES,
        BATCH,
        args.x64,
        dict(n_layers=N_LAYERS, width_one=WIDTH_ONE, width_two=WIDTH_TWO),
    )


if __name__ == '__main__':
    main()
