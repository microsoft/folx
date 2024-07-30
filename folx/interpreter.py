import functools
import logging
from collections import defaultdict
from typing import Callable, ParamSpec, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import core
from jax._src.source_info_util import summarize
from jax.typing import ArrayLike
from jax.util import safe_map

from .api import (
    IS_LEAF,
    IS_LPL_ARR,
    Array,
    ArrayOrFwdLaplArray,
    FwdJacobian,
    FwdLaplArray,
    PyTree,
)
from .utils import LoggingPrefix, extract_jacobian_mask, ravel
from .wrapped_functions import get_laplacian, wrap_forward_laplacian

R = TypeVar('R', bound=PyTree[Array])
P = ParamSpec('P')


class JaxExprEnvironment:
    # A simple environment that keeps track of the variables
    # and frees them once they are no longer needed.
    env: dict[core.Var, ArrayOrFwdLaplArray]
    reference_counter: dict[core.Var, int]

    def __init__(
        self, jaxpr: core.Jaxpr, consts: Sequence[Array], *args: ArrayOrFwdLaplArray
    ):
        self.env = {}
        self.reference_counter = defaultdict(int)
        for v in jaxpr.invars + jaxpr.constvars:
            if isinstance(v, core.Literal):
                continue
            self.reference_counter[v] += 1
        eqn: core.JaxprEqn
        for eqn in jaxpr.eqns:
            for v in eqn.invars:
                if isinstance(v, core.Literal):
                    continue
                self.reference_counter[v] += 1
        for v in jaxpr.outvars:
            if isinstance(v, core.Literal):
                continue
            self.reference_counter[v] = np.iinfo(np.int32).max
        self.write_many(jaxpr.constvars, consts)
        self.write_many(jaxpr.invars, args)

    def read(self, var: core.Atom) -> ArrayOrFwdLaplArray:
        if isinstance(var, core.Literal):
            return var.val
        self.reference_counter[var] -= 1
        result = self.env[var]
        if self.reference_counter[var] == 0:
            del self.env[var]
            del self.reference_counter[var]
        return result

    def write(self, var: core.Var, val: ArrayOrFwdLaplArray):
        if self.reference_counter[var] > 0:
            self.env[var] = val

    def read_many(self, vars: Sequence[core.Atom]) -> list[ArrayOrFwdLaplArray]:
        return safe_map(self.read, vars)

    def write_many(self, vars: Sequence[core.Var], vals: Sequence[ArrayOrFwdLaplArray]):
        return safe_map(self.write, vars, vals)


def eval_jaxpr_with_forward_laplacian(
    jaxpr: core.Jaxpr, consts, *args, sparsity_threshold: int
):
    enable_sparsity = sparsity_threshold > 0
    env = JaxExprEnvironment(jaxpr, consts, *args)

    def eval_scan(eqn: core.JaxprEqn, invals):
        n_carry, n_const = eqn.params['num_carry'], eqn.params['num_consts']
        in_const, in_carry, in_inp = (
            invals[:n_const],
            invals[n_const : n_carry + n_const],
            invals[n_const + n_carry :],
        )
        carry_merge = extract_jacobian_mask(in_carry)
        assert all(
            isinstance(x, Array) for x in in_inp
        ), 'Scan does not support scanning over input depenedent tensors.\nPlease unroll the loop.'

        def wrapped(carry, x):
            result = eval_jaxpr_with_forward_laplacian(
                eqn.params['jaxpr'].jaxpr,
                (),
                *in_const,
                *carry_merge(carry),
                *x,
                sparsity_threshold=sparsity_threshold,
            )
            return result[:n_carry], result[n_carry:]

        first_carry, first_y = wrapped(in_carry, jtu.tree_map(lambda x: x[0], in_inp))
        # Check whether jacobian sparsity matches
        for a, b in zip(in_carry, first_carry):
            if not isinstance(a, type(b)):
                raise TypeError(f'Type mismatch in scan: {type(a)} != {type(b)}')
            if isinstance(a, FwdLaplArray):
                if not np.all(a.jacobian.x0_idx == b.jacobian.x0_idx):  # type: ignore
                    raise ValueError('Jacobian sparsity mismatch in scan.')
        carry, y = jax.lax.scan(
            wrapped,  # type: ignore
            in_carry,
            in_inp,
            length=eqn.params['length'],
            reverse=eqn.params['reverse'],
            unroll=eqn.params['unroll'],
        )
        carry = [
            a._replace(jacobian=a.jacobian._replace(x0_idx=b.jacobian.x0_idx))  # type: ignore
            if isinstance(a, FwdLaplArray)
            else a
            for a, b in zip(carry, first_carry)
        ]
        y = [
            a._replace(jacobian=a.jacobian._replace(x0_idx=b.jacobian.x0_idx))  # type: ignore
            if isinstance(a, FwdLaplArray)
            else a
            for a, b in zip(y, first_y)
        ]
        return *carry, *y

    def eval_pjit(eqn: core.JaxprEqn, invals):
        name = eqn.params['name']
        if fn := get_laplacian(name):
            # TODO: this is a bit incomplete, e.g., kwargs?
            outvals = fn(invals, {}, sparsity_threshold=sparsity_threshold)
            if isinstance(outvals, (FwdLaplArray, Array)):
                outvals = [outvals]  # TODO: Figure out how to properly handle outvals
            return outvals
        sub_expr: core.ClosedJaxpr = eqn.params['jaxpr']
        return eval_jaxpr_with_forward_laplacian(
            sub_expr.jaxpr,
            sub_expr.literals,
            *invals,
            sparsity_threshold=sparsity_threshold,
        )

    def eval_custom_jvp(eqn: core.JaxprEqn, invals):
        subfuns, args = eqn.primitive.get_bind_params(eqn.params)
        fn = functools.partial(eqn.primitive.bind, *subfuns, **args)
        with LoggingPrefix(f'({summarize(eqn.source_info)})'):
            return wrap_forward_laplacian(fn)(
                invals, {}, sparsity_threshold=sparsity_threshold
            )

    def eval_laplacian(eqn: core.JaxprEqn, invals):
        subfuns, params = eqn.primitive.get_bind_params(eqn.params)
        with LoggingPrefix(f'({summarize(eqn.source_info)})'):
            fn = get_laplacian(eqn.primitive, True)
            return fn(
                (*subfuns, *invals), params, sparsity_threshold=sparsity_threshold
            )

    for eqn in jaxpr.eqns:
        invals = env.read_many(eqn.invars)
        # Eval expression
        try:
            if all(not isinstance(x, FwdLaplArray) for x in invals):
                subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
                # If non of the inputs were dependent on an FwdLaplArray,
                # we can just use the regular primitive. This will avoid
                # omnistaging. While this could cost us some memory and speed,
                # it gives us access to more variables during tracing.
                # https://github.com/google/jax/pull/3370
                if (
                    all(not isinstance(x, core.Tracer) for x in invals)
                    and enable_sparsity
                ):
                    try:
                        with jax.ensure_compile_time_eval():
                            outvals = eqn.primitive.bind(
                                *subfuns, *invals, **bind_params
                            )
                    except Exception as e:
                        with LoggingPrefix(f'({summarize(eqn.source_info)})'):
                            logging.warning(
                                f'Could not perform operation {eqn.primitive.name} in eager execution despite it only depending on non-input dependent values. '
                                'We switch to tracing rather than eager execution. This may impact sparsity propagation.\n'
                                f'{e}'
                            )
                        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
                else:
                    outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
            elif eqn.primitive.name == 'scan':
                outvals = eval_scan(eqn, invals)
            elif eqn.primitive.name == 'pjit':
                outvals = eval_pjit(eqn, invals)
            elif eqn.primitive.name == 'custom_jvp_call':
                outvals = eval_custom_jvp(eqn, invals)
            else:
                outvals = eval_laplacian(eqn, invals)
        except Exception as e:
            with LoggingPrefix(f'({summarize(eqn.source_info)})'):
                logging.error(f'Error in operation {eqn.primitive.name}.')
            raise e

        # unify output
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        # save output
        env.write_many(eqn.outvars, outvals)  # type: ignore
    return env.read_many(jaxpr.outvars)


def init_forward_laplacian_state(
    *x: PyTree[Array], sparsity: bool, weights: PyTree[ArrayLike]
) -> PyTree[FwdLaplArray]:
    """
    Initialize forward Laplacian state from a PyTree of arrays.
    """
    if any(IS_LPL_ARR(x_) for x_ in jtu.tree_leaves(x, is_leaf=IS_LEAF)):
        return x
    x_flat, unravel = ravel(x)
    if weights is None:
        weights = 1.0

    def init_weights(weight, tree):
        return jtu.tree_map(
            functools.partial(jnp.full_like, fill_value=weight),
            tree,
        )

    jac = jtu.tree_map(init_weights, weights, x)
    jac_idx = unravel(np.arange(x_flat.shape[0]))
    if jnp.iscomplexobj(x_flat):
        logging.info(
            '[folx] Found complex input. This is not well supported, results might be wrong.'
        )
    if sparsity:
        jacobian = jtu.tree_map(FwdJacobian, jac, jac_idx)
        jacobian = jtu.tree_map(lambda x: x[None], jacobian)
    else:
        jacobian = jax.vmap(unravel)(jnp.diag(ravel(jac)[0]))
        jacobian = jtu.tree_map(FwdJacobian.from_dense, jacobian)
    lapl_x = jtu.tree_map(jnp.zeros_like, x)
    return jtu.tree_map(FwdLaplArray, x, jacobian, lapl_x)


def forward_laplacian(
    fn: Callable[P, PyTree[Array]],
    sparsity_threshold: int | float = 0,
    disable_jit: bool = False,
) -> Callable[P, PyTree[FwdLaplArray]]:
    """
    This function takes a function and returns a function that computes the Laplacian of the function.
    The returned function will be jitted by default as running it in eager execution will typically be a lot slower.

    Args:
        - fn: function to compute the Laplacian of
        - sparsity_threshold: threshold for sparsity propagation.
            If the number of non-zero elements in the input is larger than this threshold,we will not propagate sparsity.
            If the value is between 0 and 1, it will be interpreted as a fraction of the total number of elements.
            If the value is larger than 1, it will be interpreted as an absolute number of elements.
            If enabling sparsity, we recommend relatively large values like 0.6 as frequent materializations are slow.
    """

    def wrapped(*args: P.args, weights: PyTree[ArrayLike] = 1.0, **kwargs: P.kwargs):
        """
        Wraps the function and computes the Laplacian.

        Args:
            - args: positional arguments to the function
            - weights: weights for the arguments to the function must be tree broadcastable (like vmap) of the input arguments
            - kwargs: keyword arguments to the function
        """
        args_arr = jtu.tree_map(
            lambda x: x.x if IS_LPL_ARR(x) else x, args, is_leaf=IS_LEAF
        )
        closed_jaxpr = jax.make_jaxpr(fn)(*args_arr, **kwargs)
        flat_args = jtu.tree_leaves(args, is_leaf=IS_LEAF)
        if 0 < sparsity_threshold < 1:
            threshold = int(sparsity_threshold * sum(x.size for x in flat_args))
        else:
            threshold = int(sparsity_threshold)
        lapl_args = init_forward_laplacian_state(
            *flat_args, sparsity=threshold > 0, weights=weights
        )
        out = eval_jaxpr_with_forward_laplacian(
            closed_jaxpr.jaxpr,
            closed_jaxpr.literals,
            *lapl_args,
            sparsity_threshold=threshold,
        )
        out_structure = jtu.tree_structure(jax.eval_shape(fn, *args_arr, **kwargs))
        return out_structure.unflatten(out)

    if disable_jit:
        return wrapped

    return jax.jit(wrapped)  # type: ignore
