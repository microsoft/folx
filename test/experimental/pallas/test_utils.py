import jax
import jax.numpy as jnp
import numpy as np
import pytest

from folx.experimental.pallas.attention.utils import dot


@pytest.mark.parametrize('trans_a', [False, True])
@pytest.mark.parametrize('trans_b', [False, True])
def test_dot(trans_a, trans_b):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    a_in = a.T if trans_a else a
    b_in = b.T if trans_b else b
    np.testing.assert_allclose(dot(a_in, b_in, trans_a, trans_b), a @ b, rtol=1e-6)


@pytest.mark.parametrize(
    'dtype,out', [(jnp.bfloat16, jnp.float32), (jnp.float64, jnp.float64)]
)
def test_dot_accumulation_dtype(dtype, out):
    x = jax.ShapeDtypeStruct((2, 2), dtype)
    assert jax.eval_shape(dot, x, x).dtype == out
