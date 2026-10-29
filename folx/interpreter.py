import functools
import logging
import warnings
from collections import defaultdict
from typing import Callable, ParamSpec, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.core import Atom, Tracer
from jax.typing import ArrayLike

from ._compat import (
    CUSTOM_VJP_PRIMITIVES,
    ClosedJaxpr,
    Jaxpr,
    JaxprEqn,
    Literal,
    Var,
    custom_vjp_primal,
    open_jaxpr,
    split_bind_params,
    summarize,
)
from .api import (
    IS_LEAF,
    IS_LPL_ARR,
    Array,
    ArrayOrFwdLaplArray,
    FwdJacobian,
    FwdLaplArray,
    PyTree,
)
from .control_flow import scan_laplacian, while_laplacian
from .utils import (
    LoggingPrefix,
    compile_time_eval,
    pvary_constants_scope,
    ravel,
    record_pvary_constant,
)
from .wrapped_functions import custom_jvp_laplacian, get_laplacian

R = TypeVar('R', bound=PyTree[Array])
P = ParamSpec('P')


class JaxExprEnvironment:
    # A simple environment that keeps track of the variables
    # and frees them once they are no longer needed.
    env: dict[Var, ArrayOrFwdLaplArray]
    reference_counter: dict[Var, int]

    def __init__(
        self, jaxpr: Jaxpr, consts: Sequence[Array], *args: ArrayOrFwdLaplArray
    ):
        self.env = {}
        self.reference_counter = defaultdict(int)
        for v in jaxpr.invars + jaxpr.constvars:
            if isinstance(v, Literal):
                continue
            self.reference_counter[v] += 1
        eqn: JaxprEqn
        for eqn in jaxpr.eqns:
            for v in eqn.invars:
                if isinstance(v, Literal):
                    continue
                self.reference_counter[v] += 1
        for v in jaxpr.outvars:
            if isinstance(v, Literal):
                continue
            self.reference_counter[v] = np.iinfo(np.int32).max
        self.write_many(jaxpr.constvars, consts)
        self.write_many(jaxpr.invars, args)

    def read(self, var: Atom) -> ArrayOrFwdLaplArray:
        if isinstance(var, Literal):
            return var.val
        self.reference_counter[var] -= 1
        result = self.env[var]
        if self.reference_counter[var] == 0:
            del self.env[var]
            del self.reference_counter[var]
        return result

    def write(self, var: Var, val: ArrayOrFwdLaplArray):
        if self.reference_counter[var] > 0:
            self.env[var] = val

    def read_many(self, vars: Sequence[Atom]) -> list[ArrayOrFwdLaplArray]:
        return list(map(self.read, vars))

    def write_many(self, vars: Sequence[Var], vals: Sequence[ArrayOrFwdLaplArray]):
        return list(map(self.write, vars, vals))


@pvary_constants_scope()
def eval_jaxpr_with_forward_laplacian(
    jaxpr: Jaxpr, consts, *args, sparsity_threshold: int
):
    enable_sparsity = sparsity_threshold > 0
    env = JaxExprEnvironment(jaxpr, consts, *args)

    def eval_sub_jaxpr(jaxpr, consts, *args):
        return eval_jaxpr_with_forward_laplacian(
            jaxpr, consts, *args, sparsity_threshold=sparsity_threshold
        )

    def eval_custom_vjp(eqn: JaxprEqn, invals):
        # A VJP rule cannot provide forward-mode derivatives, so we
        # differentiate the primal function instead.
        primal, consts, _ = custom_vjp_primal(eqn)
        return eval_sub_jaxpr(primal, consts, *invals)

    def eval_pjit(eqn: JaxprEqn, invals):
        name = eqn.params['name']
        if fn := get_laplacian(name):
            # TODO: this is a bit incomplete, e.g., kwargs?
            outvals = fn(invals, {}, sparsity_threshold=sparsity_threshold)
            if isinstance(outvals, (FwdLaplArray, Array)):
                outvals = [outvals]  # TODO: Figure out how to properly handle outvals
            return outvals
        sub_expr: ClosedJaxpr = eqn.params['jaxpr']
        return eval_jaxpr_with_forward_laplacian(
            sub_expr.jaxpr,
            sub_expr.literals,
            *invals,
            sparsity_threshold=sparsity_threshold,
        )

    def eval_custom_jvp(eqn: JaxprEqn, invals):
        subfuns, args = split_bind_params(eqn.primitive, eqn.params)
        fn = functools.partial(eqn.primitive.bind, *subfuns, **args)
        primal = open_jaxpr(eqn.params['call_jaxpr'])[0]
        with LoggingPrefix(f'({summarize(eqn.source_info)})'):
            return custom_jvp_laplacian(fn, primal, invals, sparsity_threshold)

    def eval_laplacian(eqn: JaxprEqn, invals):
        subfuns, params = split_bind_params(eqn.primitive, eqn.params)
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
                subfuns, bind_params = split_bind_params(eqn.primitive, eqn.params)
                # If non of the inputs were dependent on an FwdLaplArray,
                # we can just use the regular primitive. This will avoid
                # omnistaging. While this could cost us some memory and speed,
                # it gives us access to more variables during tracing.
                # https://github.com/google/jax/pull/3370
                if (
                    eqn.primitive.name == 'pvary'
                    and enable_sparsity
                    and all(not isinstance(x, Tracer) for x in invals)
                ):
                    # shard_map's type-only cast of a constant cannot run
                    # eagerly. Trace it and remember the constant.
                    outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
                    outs = outvals if eqn.primitive.multiple_results else [outvals]
                    for out, val in zip(outs, invals):
                        record_pvary_constant(out, val)
                elif all(not isinstance(x, Tracer) for x in invals) and enable_sparsity:
                    try:
                        with compile_time_eval():
                            outvals = eqn.primitive.bind(
                                *subfuns, *invals, **bind_params
                            )
                    except Exception as e:
                        with LoggingPrefix(f'({summarize(eqn.source_info)})'):
                            warnings.warn(
                                f'Could not perform operation {eqn.primitive.name} in eager execution despite it only depending on non-input dependent values. '
                                'We switch to tracing rather than eager execution. This may impact sparsity propagation.\n'
                                f'{e}'
                            )
                        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
                else:
                    outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
            elif eqn.primitive.name == 'scan':
                outvals = scan_laplacian(eqn, invals, eval_sub_jaxpr)
            elif eqn.primitive.name in ('jit', 'pjit'):
                outvals = eval_pjit(eqn, invals)
            elif eqn.primitive.name == 'while':
                outvals = while_laplacian(eqn, invals, eval_sub_jaxpr)
            elif eqn.primitive.name == 'custom_jvp_call':
                outvals = eval_custom_jvp(eqn, invals)
            elif eqn.primitive.name in CUSTOM_VJP_PRIMITIVES:
                outvals = eval_custom_vjp(eqn, invals)
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
        return wrapped  # type: ignore

    return jax.jit(wrapped)  # type: ignore
