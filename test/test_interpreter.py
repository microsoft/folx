import jax
import jax.numpy as jnp
import pytest

import folx._compat as compat_mod


class _DummyPrimitive:
    def __init__(self, result):
        self.result = result

    def get_bind_params(self, params):
        assert params == {'x': 1}
        return self.result


def test_split_bind_params_accepts_legacy_tuple(monkeypatch):
    monkeypatch.setattr(compat_mod, '_USES_DICT_BIND_PARAMS', False)
    subfuns, params = compat_mod.split_bind_params(
        _DummyPrimitive((['fn'], {'x': 2})), {'x': 1}
    )
    assert subfuns == ['fn']
    assert params == {'x': 2}


def test_split_bind_params_accepts_new_dict_only_api(monkeypatch):
    monkeypatch.setattr(compat_mod, '_USES_DICT_BIND_PARAMS', True)
    subfuns, params = compat_mod.split_bind_params(_DummyPrimitive({'x': 2}), {'x': 1})
    assert subfuns == ()
    assert params == {'x': 2}


def test_split_bind_params_rejects_invalid_tuple_shape(monkeypatch):
    monkeypatch.setattr(compat_mod, '_USES_DICT_BIND_PARAMS', False)
    with pytest.raises(TypeError, match='2-tuple'):
        compat_mod.split_bind_params(_DummyPrimitive(({'x': 2},)), {'x': 1})


def test_split_bind_params_rejects_dict_on_legacy_jax(monkeypatch):
    monkeypatch.setattr(compat_mod, '_USES_DICT_BIND_PARAMS', False)
    with pytest.raises(TypeError, match='JAX < 0.9.2'):
        compat_mod.split_bind_params(_DummyPrimitive({'x': 2}), {'x': 1})


def test_split_bind_params_rejects_tuple_on_new_jax(monkeypatch):
    monkeypatch.setattr(compat_mod, '_USES_DICT_BIND_PARAMS', True)
    with pytest.raises(TypeError, match='JAX >= 0.9.2'):
        compat_mod.split_bind_params(_DummyPrimitive((['fn'], {'x': 2})), {'x': 1})


def test_version_key():
    assert compat_mod.version_key('0.10.0rc1') == (0, 10, 0)
    assert compat_mod.version_key('0.9.2') < compat_mod.version_key('0.10.0')


@pytest.mark.parametrize(
    'params',
    [{'num_consts': 1, 'num_carry': 2}, {'ft_in': ((None,), (None, None), (None,))}],
)
def test_scan_num_consts_carry(params):
    assert compat_mod.scan_num_consts_carry(params) == (1, 2)


@pytest.mark.parametrize(
    'f,expected',
    [
        (lambda x: jax.lax.scan(lambda c, a: (c * a + x, c), x, jnp.ones(3)), (1, 1)),
        (lambda x: jax.lax.fori_loop(0, 3, lambda i, c: c * 2, x), (0, 2)),
    ],
)
def test_scan_num_consts_carry_matches_installed_jax(f, expected):
    eqn = jax.make_jaxpr(f)(1.0).jaxpr.eqns[-1]
    assert compat_mod.scan_num_consts_carry(eqn.params) == expected


def test_custom_vjp_primal():
    @jax.custom_vjp
    def f(x):
        return jnp.sin(x)

    f.defvjp(lambda x: (f(x), x), lambda x, g: (g,))
    (eqn,) = jax.make_jaxpr(f)(1.0).jaxpr.eqns
    assert eqn.primitive.name in compat_mod.CUSTOM_VJP_PRIMITIVES
    primal, consts, _ = compat_mod.custom_vjp_primal(eqn)
    assert jax.core.eval_jaxpr(primal, consts, 0.5)[0] == jnp.sin(0.5)
