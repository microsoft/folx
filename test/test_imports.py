import importlib
import pkgutil

import pytest

import folx

MODULES = [m.name for m in pkgutil.walk_packages(folx.__path__, 'folx.')]


@pytest.mark.parametrize('name', MODULES)
def test_import(name):
    importlib.import_module(name)
