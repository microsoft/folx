import functools
import logging
from collections import defaultdict
from typing import Callable, ParamSpec, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import core
from jax.util import safe_map

from .api import Array, ArrayOrFwdLaplArray, FunctionFlags, FwdLaplArray, PyTree
from .fwd_laplacian import add_forward_laplacian, init_forward_laplacian_state, non_lapl_call

R = TypeVar("R", bound=PyTree[Array])
P = ParamSpec("P")


@functools.partial(add_forward_laplacian, flags=FunctionFlags.INDEXING)
def rearrange(x, contract_dims, batch_dims, brdcast_dims, other_brdcast_dims, rhs=False):
    new_dims_index = (..., *([None] * len(other_brdcast_dims)))
    x_xtd = x[new_dims_index]
    new_dims = tuple(range(x.ndim, x.ndim + len(other_brdcast_dims)))
    # According to the XLA docs
    # https://www.tensorflow.org/xla/operation_semantics#dotgeneral
    # the output will be *batch_dims, *lhs_brdcast_dims, *rhs_brdcast_dims
    if rhs:
        new_dims, brdcast_dims = brdcast_dims, new_dims
    x_rearranged = x_xtd.transpose(*batch_dims, *brdcast_dims, *new_dims, *contract_dims)
    return x_rearranged.reshape(*x_rearranged.shape[: -len(contract_dims)], -1)


def dot_general(
    lhs: ArrayOrFwdLaplArray,
    rhs: ArrayOrFwdLaplArray,
    dimension_numbers: tuple[
        tuple[tuple[int, ...], tuple[int, ...]], tuple[tuple[int, ...], tuple[int, ...]]
    ],
    precision=None,
    preferred_element_type=None,
    sparsity_threshold: int = 0,
) -> ArrayOrFwdLaplArray:
    # If we have regular arrays just do regular dot_general
    if not isinstance(lhs, FwdLaplArray) and not isinstance(rhs, FwdLaplArray):
        return jax.lax.dot_general_p.bind(lhs, rhs, dimension_numbers, precision, preferred_element_type)  # type: ignore

    # So the idea for the dot product is to rearrange the arrays such that
    # the contract_dims are at the end. Then we just have to worry about
    # contracting the last dimension.

    lh_dims = tuple(range(lhs.ndim))
    rh_dims = tuple(range(rhs.ndim))
    lh_contract, rh_contract = dimension_numbers[0]
    lh_batch_dims, rh_batch_dims = dimension_numbers[1]
    lh_brdcast_dims = tuple(i for i in lh_dims if i not in lh_batch_dims + lh_contract)
    rh_brdcast_dims = tuple(i for i in rh_dims if i not in rh_batch_dims + rh_contract)

    left_inp = rearrange(
        lhs,
        contract_dims=lh_contract,
        batch_dims=lh_batch_dims,
        brdcast_dims=lh_brdcast_dims,
        other_brdcast_dims=rh_brdcast_dims,
        sparsity_threshold=sparsity_threshold,
    )
    right_inp = rearrange(
        rhs,
        contract_dims=rh_contract,
        batch_dims=rh_batch_dims,
        brdcast_dims=rh_brdcast_dims,
        other_brdcast_dims=lh_brdcast_dims,
        rhs=True,
        sparsity_threshold=sparsity_threshold,
    )

    # this einsum is somewhat inefficient.
    # one should think about rewriting the hessian
    # computation and just use the regular dot product.
    def dot_last(lhs: Array, rhs: Array) -> Array:
        return jnp.einsum(
            "...i,...i->...",
            lhs,
            rhs,
            precision=precision,
            # This flag only exists in newer JAX versions.
            # preferred_element_type=preferred_element_type
        )

    result = add_forward_laplacian(dot_last, flags=FunctionFlags.DOT_PRODUCT, in_axes=-1)(
        left_inp, right_inp, sparsity_threshold=sparsity_threshold
    )
    return result


def dtype_conversion(arr, new_dtype, sparsity_threshold=None, **kwargs):
    # we only need to keep the jacobian if it's a float type
    if new_dtype in (jnp.float16, jnp.float32, jnp.float64, jnp.complex64, jnp.complex128):
        return add_forward_laplacian(jax.lax.convert_element_type_p.bind, in_axes=())(
            arr, new_dtype=new_dtype, **kwargs
        )
    return non_lapl_call(jax.lax.convert_element_type_p.bind)(arr, new_dtype=new_dtype, **kwargs)


@jax.custom_jvp
def slogdet(x):
    # We only need this custom slog det to avoid a jax bug
    # https://github.com/google/jax/issues/17379
    return jnp.linalg.slogdet(x)


def slogdet_jvp(primals, tangents):
    # we know that the input will be a single tensor where the last two dims will be reduced
    # So, instead of using the JVP from JAX, we compute the jacobian explicitly via backprop
    # and then compute the dot product with the tangent.
    primals, tangents = primals[0], tangents[0]
    batch_shape = primals.shape[:-2]  # the last two will be reduced
    tangents = tangents.reshape(-1, *tangents.shape[-2:])
    primals = primals.reshape(-1, *primals.shape[-2:])

    sign, logdet = jnp.linalg.slogdet(primals)
    y = sign, logdet

    jacobians = jnp.linalg.inv(primals)

    def custom_jvp(jacobian, tangent):
        return (jnp.zeros(()), jnp.vdot(jacobian.T, tangent))

    y_tangent = jax.vmap(custom_jvp)(jacobians, tangents)

    y, y_tangent = jtu.tree_map(lambda x: x.reshape(*batch_shape), (y, y_tangent))
    return y, y_tangent


slogdet.defjvp(slogdet_jvp)


def slogdet_wrapper(x, *_, **__):
    sign, logdet = add_forward_laplacian(slogdet, in_axes=None, flags=FunctionFlags.SLOGDET)(
        x, sparsity_threshold=0
    )
    # Remove the jacobian of the sign
    sign = non_lapl_call(lambda x: x)(sign)
    return sign, logdet


_LAPLACE_FN_REGISTRY = {
    jax.lax.dot_general_p: dot_general,
    jax.lax.abs_p: add_forward_laplacian(jax.lax.abs, flags=FunctionFlags.LINEAR, in_axes=()),
    jax.lax.neg_p: add_forward_laplacian(jax.lax.neg, flags=FunctionFlags.LINEAR, in_axes=()),
    jax.lax.add_p: add_forward_laplacian(jax.lax.add, flags=FunctionFlags.LINEAR, in_axes=()),
    jax.lax.sub_p: add_forward_laplacian(jax.lax.sub, flags=FunctionFlags.LINEAR, in_axes=()),
    jax.lax.mul_p: add_forward_laplacian(jax.lax.mul, flags=FunctionFlags.DOT_PRODUCT, in_axes=()),
    jax.lax.div_p: add_forward_laplacian(
        jax.lax.div, flags=FunctionFlags.LINEAR_IN_FIRST, in_axes=()
    ),
    jax.lax.pow_p: add_forward_laplacian(jax.lax.pow, in_axes=()),
    jax.lax.integer_pow_p: add_forward_laplacian(jax.lax.integer_pow, in_axes=()),
    jax.lax.sign_p: non_lapl_call(jax.lax.sign),
    jax.lax.reduce_sum_p: add_forward_laplacian(
        jax.lax.reduce_sum_p.bind,
        flags=FunctionFlags.REDUCTION | FunctionFlags.LINEAR,
        name="reduce_sum",
    ),
    jax.lax.reduce_max_p: add_forward_laplacian(
        jax.lax.reduce_max_p.bind,
        flags=FunctionFlags.REDUCTION | FunctionFlags.LINEAR,
        name="reduce_max",
    ),
    jax.lax.reduce_min_p: add_forward_laplacian(
        jax.lax.reduce_min_p.bind,
        flags=FunctionFlags.REDUCTION | FunctionFlags.LINEAR,
        name="reduce_min",
    ),
    jax.lax.reduce_prod_p: add_forward_laplacian(
        jax.lax.reduce_prod_p.bind,
        flags=FunctionFlags.REDUCTION | FunctionFlags.LINEAR,
        name="reduce_prod",
    ),
    jax.lax.cumsum_p: add_forward_laplacian(jax.lax.cumsum, flags=FunctionFlags.LINEAR),
    jax.lax.sqrt_p: add_forward_laplacian(jax.lax.sqrt, in_axes=()),
    jax.lax.rsqrt_p: add_forward_laplacian(jax.lax.rsqrt, in_axes=()),
    jax.lax.log_p: add_forward_laplacian(jax.lax.log, in_axes=()),
    jax.lax.log1p_p: add_forward_laplacian(jax.lax.log1p, in_axes=()),
    jax.lax.exp_p: add_forward_laplacian(jax.lax.exp, in_axes=()),
    jax.lax.expm1_p: add_forward_laplacian(jax.lax.expm1, in_axes=()),
    jax.lax.tanh_p: add_forward_laplacian(jax.lax.tanh, in_axes=()),
    jax.lax.logistic_p: add_forward_laplacian(jax.lax.logistic, in_axes=()),
    jax.lax.acos_p: add_forward_laplacian(jax.lax.acos, in_axes=()),
    jax.lax.asin_p: add_forward_laplacian(jax.lax.asin, in_axes=()),
    jax.lax.atan_p: add_forward_laplacian(jax.lax.atan, in_axes=()),
    jax.lax.atan2_p: add_forward_laplacian(jax.lax.atan2, in_axes=()),
    jax.lax.cos_p: add_forward_laplacian(jax.lax.cos, in_axes=()),
    jax.lax.sin_p: add_forward_laplacian(jax.lax.sin, in_axes=()),
    jax.lax.tan_p: add_forward_laplacian(jax.lax.tan, in_axes=()),
    jax.lax.broadcast_in_dim_p: add_forward_laplacian(
        jax.lax.broadcast_in_dim, flags=FunctionFlags.INDEXING
    ),
    jax.lax.reshape_p: add_forward_laplacian(jax.lax.reshape, flags=FunctionFlags.INDEXING),
    jax.lax.slice_p: add_forward_laplacian(jax.lax.slice, flags=FunctionFlags.INDEXING),
    jax.lax.dynamic_slice_p: add_forward_laplacian(
        jax.lax.dynamic_slice_p.bind,
        flags=FunctionFlags.INDEXING,
        name="slice",
        index_static_args=slice(1, None),
    ),
    jax.lax.concatenate_p: add_forward_laplacian(
        jax.lax.concatenate_p.bind,
        flags=FunctionFlags.INDEXING,
        name="concatenate",
        index_static_args=(),
    ),
    jax.lax.select_n_p: add_forward_laplacian(
        jax.lax.select_n, flags=FunctionFlags.INDEXING, index_static_args=(0,)
    ),
    jax.lax.gather_p: add_forward_laplacian(
        jax.lax.gather_p.bind, flags=FunctionFlags.INDEXING, name="gather"
    ),
    jax.lax.transpose_p: add_forward_laplacian(jax.lax.transpose, flags=FunctionFlags.INDEXING),
    jax.lax.squeeze_p: add_forward_laplacian(jax.lax.squeeze, flags=FunctionFlags.INDEXING),
    jax.lax.rev_p: add_forward_laplacian(jax.lax.rev, flags=FunctionFlags.INDEXING),
    jax.lax.max_p: add_forward_laplacian(jax.lax.max, in_axes=(), flags=FunctionFlags.LINEAR),
    jax.lax.min_p: add_forward_laplacian(jax.lax.min, in_axes=(), flags=FunctionFlags.LINEAR),
    jax.lax.scatter_p: add_forward_laplacian(
        jax.lax.scatter_p.bind, flags=FunctionFlags.INDEXING | FunctionFlags.SCATTER, name="scatter"
    ),
    jax.lax.scatter_add_p: add_forward_laplacian(
        jax.lax.scatter_add_p.bind,
        flags=FunctionFlags.LINEAR | FunctionFlags.SCATTER,
        name="scatter_add",
    ),
    jax.lax.stop_gradient_p: non_lapl_call(jax.lax.stop_gradient),
    jax.lax.eq_p: non_lapl_call(jax.lax.eq),
    jax.lax.lt_p: non_lapl_call(jax.lax.lt),
    jax.lax.le_p: non_lapl_call(jax.lax.le),
    jax.lax.gt_p: non_lapl_call(jax.lax.gt),
    jax.lax.ge_p: non_lapl_call(jax.lax.ge),
    jax.lax.ne_p: non_lapl_call(jax.lax.ne),
    jax.lax.xor_p: non_lapl_call(jax.lax.bitwise_xor),
    jax.lax.not_p: non_lapl_call(jax.lax.bitwise_not),
    jax.lax.and_p: non_lapl_call(jax.lax.bitwise_and),
    jax.lax.or_p: non_lapl_call(jax.lax.bitwise_or),
    jax.lax.is_finite_p: non_lapl_call(jax.lax.is_finite),
    jax.lax.convert_element_type_p: dtype_conversion,
    "sign": add_forward_laplacian(jnp.sign, flags=FunctionFlags.LINEAR),
    "logaddexp": add_forward_laplacian(jnp.logaddexp, in_axes=()),
    "sigmoid": add_forward_laplacian(jax.nn.sigmoid, in_axes=()),
    "softplus": add_forward_laplacian(jax.nn.softplus, in_axes=()),
    "silu": add_forward_laplacian(jax.nn.silu, in_axes=()),
    "slogdet": slogdet_wrapper,
}


class JaxExprEnvironment:
    # A simple environment that keeps track of the variables
    # and frees them once they are no longer needed.
    env: dict[core.Var, ArrayOrFwdLaplArray]
    reference_counter: dict[core.Var, int]

    def __init__(self, jaxpr: core.Jaxpr, consts: Sequence[Array], *args: ArrayOrFwdLaplArray):
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


def eval_jaxpr_with_forward_laplacian(jaxpr: core.Jaxpr, consts, *args, sparsity_threshold: int):
    enable_sparsity = sparsity_threshold > 0
    env = JaxExprEnvironment(jaxpr, consts, *args)

    def eval_pjit(eqn: core.JaxprEqn, invals):
        name = eqn.params["name"]
        if name in _LAPLACE_FN_REGISTRY:
            # TODO: this is a bit incomplete, e.g., kwargs?
            fn = _LAPLACE_FN_REGISTRY[name]
            outvals = fn(*invals, sparsity_threshold=sparsity_threshold)
            if isinstance(outvals, (FwdLaplArray, Array)):
                outvals = [outvals]  # TODO: Figure out how to properly handle outvals
            return outvals
        else:
            sub_expr: core.ClosedJaxpr = eqn.params["jaxpr"]
            return eval_jaxpr_with_forward_laplacian(
                sub_expr.jaxpr, sub_expr.literals, *invals, sparsity_threshold=sparsity_threshold
            )

    def eval_missing(eqn: core.JaxprEqn, invals):
        logging.warning(
            f"{eqn.primitive} not in registry. The following call might be slow as we will compute the full hessian."
        )
        return add_forward_laplacian(eqn.primitive.bind)(
            *invals, **eqn.params, sparsity_threshold=sparsity_threshold
        )

    def eval_registry(eqn: core.JaxprEqn, invals):
        fn = _LAPLACE_FN_REGISTRY[eqn.primitive]
        return fn(*invals, **eqn.params, sparsity_threshold=sparsity_threshold)

    for eqn in jaxpr.eqns:
        invals = env.read_many(eqn.invars)
        # Eval expression
        if all(not isinstance(x, FwdLaplArray) for x in invals):
            subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            # If non of the inputs were dependent on an FwdLaplArray,
            # we can just use the regular primitive. This will avoid
            # omnistaging. While this could cost us some memory and speed,
            # it gives us access to more variables during tracing.
            # https://github.com/google/jax/pull/3370
            if all(not isinstance(x, core.Tracer) for x in invals) and enable_sparsity:
                try:
                    with core.new_main(core.EvalTrace, dynamic=True):
                        outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
                except Exception as e:
                    logging.warning(
                        f"Could not perform operation {eqn.primitive.name} in eager execution despite it only depending on non-input dependent values. "
                        "We switch to tracing rather than eager execution. This may impact sparsity propagation.\n"
                        f"{e}"
                    )
                    outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
            else:
                outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
        elif eqn.primitive.name == "pjit":
            outvals = eval_pjit(eqn, invals)
        elif eqn.primitive not in _LAPLACE_FN_REGISTRY:
            outvals = eval_missing(eqn, invals)
        else:
            outvals = eval_registry(eqn, invals)

        # unify output
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        # save output
        env.write_many(eqn.outvars, outvals)  # type: ignore
    return env.read_many(jaxpr.outvars)


def forward_laplacian(
    fn: Callable[P, PyTree[Array]],
    sparsity_threshold: int | float = 0,
) -> Callable[P, PyTree[FwdLaplArray]]:
    """
    This function takes a function and returns a function that computes the Laplacian of the function.

    Args:
        - fn: function to compute the Laplacian of
        - sparsity_threshold: threshold for sparsity propagation.
            If the number of non-zero elements in the input is larger than this threshold,we will not propagate sparsity.
            If the value is between 0 and 1, it will be interpreted as a fraction of the total number of elements.
            If the value is larger than 1, it will be interpreted as an absolute number of elements.
            If enabling sparsity, we recommend relatively large values like 0.6 as frequent materializations are slow.
    """

    def wrapped(*args: P.args, **kwargs: P.kwargs):
        closed_jaxpr = jax.make_jaxpr(fn)(*args, **kwargs)
        flat_args = jtu.tree_leaves(args)
        if 0 < sparsity_threshold < 1:
            threshold = int(sparsity_threshold * sum(x.size for x in flat_args))
        else:
            threshold = int(sparsity_threshold)
        lapl_args = init_forward_laplacian_state(*flat_args, sparsity=threshold > 0)
        out = eval_jaxpr_with_forward_laplacian(
            closed_jaxpr.jaxpr, closed_jaxpr.literals, *lapl_args, sparsity_threshold=threshold
        )
        if len(out) == 1:
            return out[0]
        return out

    return wrapped
