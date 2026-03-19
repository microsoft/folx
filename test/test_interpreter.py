import pytest

import folx.interpreter as interpreter_mod


class _DummyPrimitive:
    def __init__(self, result):
        self.result = result

    def get_bind_params(self, params):
        assert params == {'x': 1}
        return self.result


def test_split_bind_params_accepts_legacy_tuple(monkeypatch):
    monkeypatch.setattr(interpreter_mod, '_USES_DICT_BIND_PARAMS', False)
    subfuns, params = interpreter_mod._split_bind_params(
        _DummyPrimitive((['fn'], {'x': 2})), {'x': 1}
    )
    assert subfuns == ['fn']
    assert params == {'x': 2}


def test_split_bind_params_accepts_new_dict_only_api(monkeypatch):
    monkeypatch.setattr(interpreter_mod, '_USES_DICT_BIND_PARAMS', True)
    subfuns, params = interpreter_mod._split_bind_params(
        _DummyPrimitive({'x': 2}), {'x': 1}
    )
    assert subfuns == ()
    assert params == {'x': 2}


def test_split_bind_params_rejects_invalid_tuple_shape(monkeypatch):
    monkeypatch.setattr(interpreter_mod, '_USES_DICT_BIND_PARAMS', False)
    with pytest.raises(TypeError, match='2-tuple'):
        interpreter_mod._split_bind_params(_DummyPrimitive(({'x': 2},)), {'x': 1})


def test_split_bind_params_rejects_dict_on_legacy_jax(monkeypatch):
    monkeypatch.setattr(interpreter_mod, '_USES_DICT_BIND_PARAMS', False)
    with pytest.raises(TypeError, match='JAX < 0.9.2'):
        interpreter_mod._split_bind_params(_DummyPrimitive({'x': 2}), {'x': 1})


def test_split_bind_params_rejects_tuple_on_new_jax(monkeypatch):
    monkeypatch.setattr(interpreter_mod, '_USES_DICT_BIND_PARAMS', True)
    with pytest.raises(TypeError, match='JAX >= 0.9.2'):
        interpreter_mod._split_bind_params(_DummyPrimitive((['fn'], {'x': 2})), {'x': 1})
