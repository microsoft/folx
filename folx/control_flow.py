"""Forward Laplacian rules for the ``scan`` and ``while`` control flow primitives."""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Atom

from ._compat import JaxprEqn, open_jaxpr, scan_num_consts_carry
from .api import JAC_DIM, Array, ArrayOrFwdLaplArray, FwdJacobian, FwdLaplArray, PyTree

EvalJaxpr = Callable[..., list[ArrayOrFwdLaplArray]]


def as_arrays(vals: Sequence[ArrayOrFwdLaplArray], vars: Sequence[Atom]) -> list:
    """Converts non-``FwdLaplArray`` values (e.g., literals) to strongly typed arrays.

    Args:
        vals: Values of the equation inputs.
        vars: Corresponding jaxpr variables or literals.

    Returns:
        List of values with plain values converted to arrays of their aval dtype.
    """
    return [
        x
        if isinstance(x, FwdLaplArray)
        else jax.lax.convert_element_type(jnp.asarray(x), v.aval.dtype)  # type: ignore
        for x, v in zip(vals, vars)
    ]


def _jac_rows(jac: FwdJacobian) -> int:
    return jac.data.shape[JAC_DIM] if jac.x0_idx is None else jac.max_n + 1


def _same_structure(a: FwdJacobian, b: FwdJacobian) -> bool:
    if a.x0_idx is None or b.x0_idx is None:
        return a.x0_idx is None and b.x0_idx is None and a.data.shape == b.data.shape
    return a.x0_idx.shape == b.x0_idx.shape and bool(np.all(a.x0_idx == b.x0_idx))


def _to_dense(x: FwdLaplArray, rows: int) -> FwdLaplArray:
    if x.jacobian.x0_idx is None:
        data = x.jacobian.data
    else:
        data = x.jacobian.materialize_for_idx(x.jacobian.x0_idx, rows)
    pad = [(0, 0)] * data.ndim
    pad[JAC_DIM] = (0, rows - data.shape[JAC_DIM])
    return FwdLaplArray(x.x, FwdJacobian(jnp.pad(data, pad)), x.laplacian)


def _cast_to(x: ArrayOrFwdLaplArray, template: ArrayOrFwdLaplArray):
    """Casts ``x`` to the type, dtype, and Jacobian structure of ``template``."""
    if not isinstance(template, FwdLaplArray):
        assert not isinstance(x, FwdLaplArray)
        return jax.lax.convert_element_type(x, template.dtype)
    t_jac = template.jacobian
    if not isinstance(x, FwdLaplArray):
        x = FwdLaplArray(x, FwdJacobian(jnp.zeros_like(t_jac.data)), jnp.zeros_like(x))
    elif t_jac.x0_idx is None:
        x = _to_dense(x, t_jac.data.shape[JAC_DIM])
    elif x.jacobian.x0_idx is not None:
        assert _same_structure(x.jacobian, t_jac)
    return FwdLaplArray(
        jax.lax.convert_element_type(x.x, template.x.dtype),
        FwdJacobian(
            jax.lax.convert_element_type(x.jacobian.data, t_jac.data.dtype),
            t_jac.x0_idx,
        ),
        jax.lax.convert_element_type(x.laplacian, template.laplacian.dtype),
    )


def _strip_idx(x: ArrayOrFwdLaplArray):
    # Sparsity indices are static and must not become part of the loop state.
    if isinstance(x, FwdLaplArray):
        return x._replace(jacobian=x.jacobian._replace(x0_idx=None))
    return x


def _stack_like(stacked: ArrayOrFwdLaplArray, step: ArrayOrFwdLaplArray):
    """Moves the scan axis of stacked outputs behind the Jacobian dimension.

    Args:
        stacked: Output of ``lax.scan`` with the scan axis leading.
        step: Output of a single traced step, defining the Jacobian sparsity.

    Returns:
        ``stacked`` with its Jacobian in folx layout.
    """
    if not isinstance(step, FwdLaplArray):
        return stacked
    assert isinstance(stacked, FwdLaplArray) and JAC_DIM == 0
    data = jnp.moveaxis(stacked.jacobian.data, 0, 1)
    idx = step.jacobian.x0_idx
    if idx is not None:
        idx = np.broadcast_to(idx[:, None], data.shape)
    return stacked._replace(jacobian=FwdJacobian(data, idx))


def loop_invariant_carry(
    carry: Sequence[ArrayOrFwdLaplArray],
    step: Callable[..., tuple[Sequence[ArrayOrFwdLaplArray], PyTree]],
    *step_args: PyTree[Array],
) -> tuple[list, Callable[[Sequence], list], Callable[[Sequence], list], PyTree]:
    """Finds a loop carry whose type and Jacobian sparsity are preserved by ``step``.

    Carries that become input dependent are promoted to ``FwdLaplArray`` and
    carries whose sparsity pattern changes are densified until a fixed point.

    Args:
        carry: Initial loop carry.
        step: Loop body mapping ``(carry, *step_args)`` to ``(new_carry, extra)``.
        step_args: Traced (abstract) arguments of a single loop step.

    Returns:
        Tuple of the initial loop state, a function restoring the carry from a
        loop state, a function converting a step's carry output to a loop
        state, and ``extra`` of the final traced step. Loop states are carries
        cast to the invariant structure without the static sparsity indices.
    """
    templates = list(carry)
    for _ in range(2 * len(templates) + 2):
        traced = {}

        def probe(c, *args):
            traced['out'] = step(restore(c), *args)
            return None

        def restore(c):
            return [_cast_to(x, t) for x, t in zip(c, templates)]

        jax.eval_shape(probe, [_strip_idx(t) for t in templates], *step_args)
        out, extra = traced['out']
        changed = False
        for i, (t, o) in enumerate(zip(templates, out)):
            if not isinstance(o, FwdLaplArray):
                continue
            if not isinstance(t, FwdLaplArray):
                o_jac = o.jacobian
                templates[i] = FwdLaplArray(
                    t,
                    FwdJacobian(
                        jnp.zeros(o_jac.data.shape, t.dtype),
                        o_jac.x0_idx,
                    ),
                    jnp.zeros_like(t),
                )
                changed = True
            elif t.jacobian.x0_idx is not None and not _same_structure(
                t.jacobian, o.jacobian
            ):
                templates[i] = _to_dense(
                    t, max(_jac_rows(t.jacobian), _jac_rows(o.jacobian))
                )
                changed = True
            elif t.jacobian.x0_idx is None and _jac_rows(o.jacobian) > _jac_rows(
                t.jacobian
            ):
                templates[i] = _to_dense(t, _jac_rows(o.jacobian))
                changed = True
        if not changed:

            def to_state(c):
                return [_strip_idx(_cast_to(x, t)) for x, t in zip(c, templates)]

            return to_state(carry), restore, to_state, extra
    raise RuntimeError('Could not find a loop invariant Jacobian structure.')


def scan_laplacian(
    eqn: JaxprEqn, invals: Sequence[ArrayOrFwdLaplArray], eval_jaxpr: EvalJaxpr
) -> list[ArrayOrFwdLaplArray]:
    """Evaluates a ``scan`` equation with forward Laplacian propagation.

    Args:
        eqn: The ``scan`` equation.
        invals: Values of its inputs.
        eval_jaxpr: Forward Laplacian interpreter ``(jaxpr, consts, *args)``.

    Returns:
        Final carry followed by the stacked outputs.
    """
    n_const, n_carry = scan_num_consts_carry(eqn.params)
    body, body_consts = open_jaxpr(eqn.params['jaxpr'])
    in_const = invals[:n_const]
    in_carry = as_arrays(invals[n_const : n_const + n_carry], eqn.invars[n_const:])
    if any(isinstance(x, FwdLaplArray) for x in invals[n_const + n_carry :]):
        raise NotImplementedError(
            'forward_laplacian does not support scanning over input dependent '
            'arrays. Please unroll the loop.'
        )
    in_xs = as_arrays(invals[n_const + n_carry :], eqn.invars[n_const + n_carry :])

    def step(carry, x):
        out = eval_jaxpr(body, body_consts, *in_const, *carry, *x)
        return out[:n_carry], out[n_carry:]

    x_slice = [jax.ShapeDtypeStruct(x.shape[1:], x.dtype) for x in in_xs]
    init, restore, to_state, ys = loop_invariant_carry(in_carry, step, x_slice)

    def body_fn(state, x):
        carry, y = step(restore(state), x)
        return to_state(carry), [_strip_idx(v) for v in y]

    state, stacked = jax.lax.scan(
        body_fn,
        init,
        in_xs,
        length=eqn.params['length'],
        reverse=eqn.params['reverse'],
        unroll=eqn.params['unroll'],
    )
    return [*restore(state), *map(_stack_like, stacked, ys)]


def while_laplacian(
    eqn: JaxprEqn, invals: Sequence[ArrayOrFwdLaplArray], eval_jaxpr: EvalJaxpr
) -> list[ArrayOrFwdLaplArray]:
    """Evaluates a ``while`` equation with forward Laplacian propagation.

    Args:
        eqn: The ``while`` equation.
        invals: Values of its inputs.
        eval_jaxpr: Forward Laplacian interpreter ``(jaxpr, consts, *args)``.

    Returns:
        Final carry.
    """
    params = eqn.params
    n_cond, n_body = params['cond_nconsts'], params['body_nconsts']
    cond, cond_consts = open_jaxpr(params['cond_jaxpr'])
    body, body_consts = open_jaxpr(params['body_jaxpr'])
    cond_args, body_args = invals[:n_cond], invals[n_cond : n_cond + n_body]
    if any(isinstance(x, FwdLaplArray) for x in cond_args):
        raise NotImplementedError(
            'forward_laplacian does not support while_loop conditions that '
            'close over input dependent values.'
        )

    def step(carry):
        return eval_jaxpr(body, body_consts, *body_args, *carry), None

    def cond_fn(state):
        carry = [x.x if isinstance(x, FwdLaplArray) else x for x in state]
        return jax.core.eval_jaxpr(cond, cond_consts, *cond_args, *carry)[0]

    in_carry = as_arrays(invals[n_cond + n_body :], eqn.invars[n_cond + n_body :])
    init, restore, to_state, _ = loop_invariant_carry(in_carry, step)
    state = jax.lax.while_loop(
        cond_fn, lambda state: to_state(step(restore(state))[0]), init
    )
    return restore(state)
