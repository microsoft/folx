import jax
import jax.numpy as jnp
import numpy as np
from folx import ForwardLaplacianOperator
from jax.experimental import enable_x64


def brute_force_laplacian(fn):
    def laplacian(x):
        H = jax.hessian(fn)(x)
        f = jax.grad(fn)(x)
        return jnp.trace(H), f
    return laplacian


def test_dense_forward_laplacian():
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    weights = jax.random.normal(subkey, (4, 256, 256)) / jnp.sqrt(256)
    key, subkey = jax.random.split(key)
    biases = jax.random.normal(subkey, (4, 256))

    def fn(x):
        y = x
        for W, b in zip(weights, biases):
            y = jnp.tanh(y @ W + b)
        return y.sum()

    bf_lapl = brute_force_laplacian(fn)
    fwd_lapl = ForwardLaplacianOperator(0)(fn)
    fwd_lapl_sparse = ForwardLaplacianOperator(0.75)(fn)
    with enable_x64():
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (256,))
        fwd_lapl, fwd_qf = fwd_lapl(x)
        sp_fwd_lapl, sp_fwd_qf = fwd_lapl_sparse(x)
        bf_lapl, bf_qf = bf_lapl(x)
        np.testing.assert_allclose(fwd_lapl, bf_lapl)
        np.testing.assert_allclose(sp_fwd_lapl, bf_lapl)
        np.testing.assert_allclose(fwd_qf, bf_qf)
        np.testing.assert_allclose(sp_fwd_qf, bf_qf)


def test_sparse_forward_laplacian():
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    weights = jax.random.normal(subkey, (4, 256, 256)) / jnp.sqrt(256)
    key, subkey = jax.random.split(key)
    biases = jax.random.normal(subkey, (4, 256))

    def fn(x):
        # MLP on a set
        y = x.reshape(10, 256)
        for W, b in zip(weights, biases):
            y = jnp.tanh(y @ W + b)
        return y.sum()

    bf_lapl = brute_force_laplacian(fn)
    fwd_lapl = ForwardLaplacianOperator(0)(fn)
    fwd_lapl_sparse = ForwardLaplacianOperator(0.75)(fn)
    with enable_x64():
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (10, 256)).reshape(-1)
        fwd_lapl, fwd_qf = fwd_lapl(x)
        sp_fwd_lapl, sp_fwd_qf = fwd_lapl_sparse(x)
        bf_lapl, bf_qf = bf_lapl(x)
        np.testing.assert_allclose(fwd_lapl, bf_lapl)
        np.testing.assert_allclose(sp_fwd_lapl, bf_lapl)
        np.testing.assert_allclose(fwd_qf, bf_qf)
        np.testing.assert_allclose(sp_fwd_qf, bf_qf)
