"""Compatibility layer for the supported range of JAX versions."""

import re
from typing import TYPE_CHECKING, Any, Sequence

import jax

if TYPE_CHECKING:
    from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Primitive, Var
    from jax.extend.source_info_util import summarize
else:
    try:
        from jax.extend.core import (
            ClosedJaxpr,
            Jaxpr,
            JaxprEqn,
            Literal,
            Primitive,
            Var,
        )
        from jax.extend.source_info_util import summarize
    except ImportError:
        from jax._src.source_info_util import summarize  # type: ignore[import-error]
        from jax.core import (  # type: ignore[import-error]
            ClosedJaxpr,
            Jaxpr,
            JaxprEqn,
            Literal,
            Primitive,
            Var,
        )

try:
    from jax.extend.core.primitives import add_jaxvals_p as add_any_p
except ImportError:
    from jax._src.ad_util import add_any_p  # type: ignore[import-error]

__all__ = [
    'CUSTOM_VJP_PRIMITIVES',
    'JAX_VERSION',
    'ClosedJaxpr',
    'Jaxpr',
    'JaxprEqn',
    'Literal',
    'Primitive',
    'Var',
    'add_any_p',
    'custom_vjp_primal',
    'mark_varying_like',
    'open_jaxpr',
    'scan_num_consts_carry',
    'split_bind_params',
    'summarize',
    'varying_axes',
    'version_key',
]


def version_key(version: str) -> tuple[int, ...]:
    """Parses a version string into a comparable tuple of its numeric parts.

    Args:
        version: Version string such as ``'0.9.2'`` or ``'0.10.0rc1'``.

    Returns:
        Tuple of integers, e.g. ``(0, 10, 0)``.
    """
    return tuple(
        int(match.group(0))
        for part in version.split('.')
        if (match := re.match(r'\d+', part)) is not None
    )


JAX_VERSION = version_key(jax.__version__)
_USES_DICT_BIND_PARAMS = JAX_VERSION >= (0, 9, 2)

# Older JAX versions name the primitive ``custom_vjp_call_jaxpr``.
CUSTOM_VJP_PRIMITIVES = ('custom_vjp_call', 'custom_vjp_call_jaxpr')


def split_bind_params(primitive, params) -> tuple[Sequence[Any], dict[str, Any]]:
    """Splits equation params into ``bind`` subfunctions and keyword params.

    JAX >= 0.9.2 returns only a dict from ``get_bind_params``; older versions
    return a ``(subfuns, params)`` tuple.

    Args:
        primitive: Primitive of the equation.
        params: Parameters of the equation.

    Returns:
        Tuple of positional subfunctions and keyword parameters for ``bind``.
    """
    bind_params = primitive.get_bind_params(params)
    if _USES_DICT_BIND_PARAMS:
        if not isinstance(bind_params, dict):
            raise TypeError(
                'Expected primitive.get_bind_params() to return a dict for '
                f'JAX >= 0.9.2, got {bind_params!r}.'
            )
        return (), bind_params
    if not isinstance(bind_params, tuple) or len(bind_params) != 2:
        raise TypeError(
            'Expected primitive.get_bind_params() to return a 2-tuple for '
            f'JAX < 0.9.2, got {bind_params!r}.'
        )
    return bind_params


def open_jaxpr(jaxpr: 'Jaxpr | ClosedJaxpr') -> tuple['Jaxpr', Sequence[Any]]:
    """Returns the open jaxpr and its constants.

    Newer JAX versions store some higher-order primitive bodies as open
    ``Jaxpr`` instead of ``ClosedJaxpr``.

    Args:
        jaxpr: Open or closed jaxpr.

    Returns:
        Tuple of the open jaxpr and the values of its ``constvars``.
    """
    if isinstance(jaxpr, ClosedJaxpr):
        return jaxpr.jaxpr, jaxpr.consts
    return jaxpr, ()


def scan_num_consts_carry(params: dict[str, Any]) -> tuple[int, int]:
    """Returns the number of constant and carry operands of a ``scan``.

    JAX >= 0.11 replaced ``num_consts``/``num_carry`` by the flat-tree
    signature ``ft_in = (consts, carry, xs)``.

    Args:
        params: Parameters of the ``scan`` equation.

    Returns:
        Tuple ``(num_consts, num_carry)``.
    """
    if 'num_carry' in params:
        return params['num_consts'], params['num_carry']
    # ``ft_in`` is a flat tree whose iteration yields leaves, not groups.
    ft_in = params['ft_in']
    consts, carry, _ = getattr(ft_in, 'elts', ft_in)
    return len(consts), len(carry)


def custom_vjp_primal(eqn: 'JaxprEqn') -> tuple['Jaxpr', Sequence[Any], int]:
    """Returns the primal computation of a ``custom_vjp`` equation.

    Args:
        eqn: Equation of a ``custom_vjp_call`` (or legacy
            ``custom_vjp_call_jaxpr``) primitive.

    Returns:
        Tuple of the open primal jaxpr, its constants, and the number of
        leading operands that are constants of the custom_vjp function.
    """
    params = eqn.params
    fun = params['call_jaxpr'] if 'call_jaxpr' in params else params['fun_jaxpr']
    jaxpr, consts = open_jaxpr(fun)
    return jaxpr, consts, params.get('num_consts', 0)


def varying_axes(x: jax.Array) -> tuple[int | str, ...]:
    """Returns the mesh axes along which ``x`` varies inside ``shard_map``.

    Args:
        x: Array or tracer.

    Returns:
        Tuple of varying mesh axis names; empty on JAX versions without
        varying-manual-axes typing.
    """
    if not hasattr(jax, 'typeof'):
        return ()

    typ = jax.typeof(x)
    manual_axis_type = getattr(typ, 'manual_axis_type', None)
    if manual_axis_type is not None:
        return tuple(getattr(manual_axis_type, 'varying', ()))

    return tuple(getattr(typ, 'vma', ()))


def mark_varying_like(x: jax.Array, like: jax.Array) -> jax.Array:
    """Marks ``x`` as varying along the same mesh axes as ``like``.

    Args:
        x: Array to mark.
        like: Reference array.

    Returns:
        ``x``, cast to vary along the mesh axes of ``like`` if needed.
    """
    axes = varying_axes(like)
    if not axes:
        return x

    if hasattr(jax.lax, 'pcast'):
        return jax.lax.pcast(x, axes, to='varying')
    if hasattr(jax.lax, 'pvary'):
        return jax.lax.pvary(x, axes)
    return x
