"""Shared helpers for the benchmarks in the folx paper."""

import gc
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import numpy as np

import folx

RESULTS_DIR = Path(__file__).parent / 'results'

METHODS = {
    'hessian_loop': 'Hessian loop',
    'folx_dense': 'folx, dense',
    'folx_sparse': 'folx, sparse',
    'folx_sparse_chunked': 'folx, sparse, chunked',
    'folx_dense_chunked': 'folx, dense, chunked',
}


def laplacian_operator(method: str, sparsity_threshold: int):
    """Returns a function mapping f to x -> (laplacian, jacobian)."""
    if method == 'hessian_loop':
        return folx.LoopLaplacianOperator()
    if method in ('folx_dense', 'folx_dense_chunked'):
        return folx.ForwardLaplacianOperator(0)
    if method in ('folx_sparse', 'folx_sparse_chunked'):
        return folx.ForwardLaplacianOperator(sparsity_threshold)
    raise ValueError(method)


def batched(method: str, fn, chunk_size: int):
    """Vectorizes fn over the batch, in sequential chunks for the chunked variants."""
    if method.endswith('_chunked'):
        return folx.batched_vmap(fn, max_batch_size=chunk_size)
    return jax.vmap(fn)


def warmup():
    """Initializes the backend so that the first benchmark does not pay for it."""
    x = jax.numpy.ones((128, 128))
    jax.block_until_ready(jax.jit(lambda x: jax.numpy.tanh(x @ x).sum())(x))


def total_laplacian(laplacian) -> np.ndarray:
    """Sums per-coordinate second derivatives, as returned by the Hessian loop, per sample."""
    laplacian = np.asarray(laplacian)
    return laplacian.reshape(laplacian.shape[0], -1).sum(-1)


def benchmark(fn, *args, n_runs: int = 20, min_time: float = 1.0) -> dict:
    """Times compilation, the first call, and steady-state calls of ``jax.jit(fn)``.

    ``compile`` includes tracing, i.e., the jaxpr interpretation of folx, lowering,
    and XLA compilation. ``run`` is the median over at least ``n_runs`` calls and
    ``min_time`` seconds. ``memory`` is the peak device memory estimated by XLA
    (arguments, outputs, and temporaries) of the compiled program.
    """
    gc.collect()
    t0 = time.perf_counter()
    compiled = jax.jit(fn).lower(*args).compile()
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = jax.block_until_ready(compiled(*args))
    t_first = time.perf_counter() - t0

    times = []
    start = time.perf_counter()
    while len(times) < n_runs or time.perf_counter() - start < min_time:
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        times.append(time.perf_counter() - t0)
        if len(times) >= 10 * n_runs:
            break

    mem = compiled.memory_analysis()
    memory = None
    if mem is not None:
        memory = (
            mem.argument_size_in_bytes
            + mem.output_size_in_bytes
            - mem.alias_size_in_bytes
            + mem.temp_size_in_bytes
        )
    return dict(
        compile=t_compile,
        first_call=t_first,
        run=statistics.median(times),
        run_std=statistics.stdev(times) if len(times) > 1 else 0.0,
        n_runs=len(times),
        memory=memory,
        output=out,
    )


def run_scaling(
    name,
    make_fn,
    n_electrons,
    methods,
    sparsity_threshold,
    chunk_sizes,
    batch,
    x64,
    config,
):
    """Benchmarks the Laplacian of ``make_fn(n, key)`` for every number of electrons.

    Inputs are Gaussian samples around a chain of atoms spaced 1.4 apart. Results,
    including the error relative to the first method, are saved to ``results/{name}.json``.
    """
    import os

    import jax.numpy as jnp

    results = {}
    for n in n_electrons:
        key_params, key_x = jax.random.split(jax.random.key(n))
        fn = make_fn(n, key_params)
        atoms = jnp.stack([jnp.arange(n) * 1.4, jnp.zeros(n), jnp.zeros(n)], -1)
        x = jax.random.normal(key_x, (batch, n * 3)) + atoms.reshape(1, -1)
        results[n], reference = {}, None
        for method in methods:
            op = laplacian_operator(method, sparsity_threshold)
            chunk_size = int(os.environ.get('FOLX_CHUNK_SIZE', chunk_sizes.get(n, 8)))
            lapl = batched(method, op(fn), chunk_size)
            try:
                r = benchmark(lapl, x, n_runs=5, min_time=0.5)
            except Exception as e:  # out of memory
                print(f'N={n:<4} {METHODS[method]:<14} failed: {str(e)[:120]}')
                results[n][method] = dict(error=str(e)[:500])
                continue
            laplacian = total_laplacian(r.pop('output')[0])
            if reference is None:
                reference = laplacian
            r['max_rel_error'] = float(
                np.max(np.abs(laplacian - reference) / np.abs(reference))
            )
            results[n][method] = r
            print(
                f'N={n:<4} {METHODS[method]:<14} compile {r["compile"]:6.2f} s, '
                f'run {r["run"] * 1e3:9.3f} ms, memory {r["memory"] / 2**20:9.1f} MiB, '
                f'rel. error {r["max_rel_error"]:.1e}',
                flush=True,
            )

    config = dict(
        config,
        batch=batch,
        sparsity_threshold=sparsity_threshold,
        chunk_sizes=chunk_sizes,
        dtype='float64' if x64 else 'float32',
    )
    save(
        f'{name}_x64' if x64 else name,
        dict(config=config, system=device_info(), results=results),
    )


def device_info() -> dict:
    device = jax.devices()[0]
    return dict(
        device=getattr(device, 'device_kind', str(device)),
        platform=device.platform,
        jax=jax.__version__,
        folx=folx.__version__,
        python=platform.python_version(),
        x64=bool(jax.config.read('jax_enable_x64')),
    )


def save(name: str, data: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f'{name}.json'
    path.write_text(json.dumps(data, indent=2, default=_to_builtin))
    print(f'Saved {path}')


def _to_builtin(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    raise TypeError(type(x))
