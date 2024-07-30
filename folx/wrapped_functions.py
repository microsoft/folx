import functools
import logging
from typing import Any, Literal, ParamSpec, TypeVar, overload

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.core import Primitive

from folx.ad import is_tree_complex

from .api import (
    Array,
    ArrayOrFwdLaplArray,
    ForwardLaplacian,
    FunctionFlags,
    FwdJacobian,
    FwdLaplArray,
    PyTree,
)
from .custom_hessian import (
    complex_abs_jac_hessian_jac,
    div_jac_hessian_jac,
    slogdet_jac_hessian_jac,
)
from .wrapper import (
    warp_without_fwd_laplacian,
    wrap_elementwise,
    wrap_forward_laplacian,
)

R = TypeVar('R', bound=PyTree[Array])
P = ParamSpec('P')


@functools.partial(wrap_forward_laplacian, flags=FunctionFlags.INDEXING)
def rearrange(
    x, contract_dims, batch_dims, brdcast_dims, other_brdcast_dims, rhs=False
):
    new_dims_index = (..., *([None] * len(other_brdcast_dims)))
    x_xtd = x[new_dims_index]
    new_dims = tuple(range(x.ndim, x.ndim + len(other_brdcast_dims)))
    # Accroding to the XLA docs
    # https://www.tensorflow.org/xla/operation_semantics#dotgeneral
    # the output will be *batch_dims, *lhs_brdcast_dims, *rhs_brdcast_dims
    if rhs:
        new_dims, brdcast_dims = brdcast_dims, new_dims
    x_rearranged = x_xtd.transpose(
        *batch_dims, *brdcast_dims, *new_dims, *contract_dims
    )
    if len(contract_dims) > 0:
        return x_rearranged.reshape(*x_rearranged.shape[: -len(contract_dims)], -1)
    # If there are no contract dims, we need to add a dummy dimensions
    return x_rearranged[..., None]


def dot_general(
    args: tuple[ArrayOrFwdLaplArray, ArrayOrFwdLaplArray],
    kwargs: dict[str, Any],
    sparsity_threshold: int = 0,
) -> ArrayOrFwdLaplArray:
    lhs, rhs = args
    dimension_numbers = kwargs['dimension_numbers']
    precision = kwargs['precision']
    preferred_element_type = kwargs['preferred_element_type']
    # If we have regular arrays just do regular dot_general
    if not isinstance(lhs, FwdLaplArray) and not isinstance(rhs, FwdLaplArray):
        return jax.lax.dot_general_p.bind(
            lhs, rhs, dimension_numbers, precision, preferred_element_type
        )  # type: ignore

    # So the idea for the dot product is to rearrange the arrays such that
    # the contract_dims are at the end. Then we just have to worry about
    # contracting the last dimension.

    lh_dims = tuple(range(lhs.ndim))
    rh_dims = tuple(range(rhs.ndim))
    lh_contract, rh_contract = dimension_numbers[0]
    lh_batch_dims, rh_batch_dims = dimension_numbers[1]
    lh_brdcast_dims = tuple(i for i in lh_dims if i not in lh_batch_dims + lh_contract)
    rh_brdcast_dims = tuple(i for i in rh_dims if i not in rh_batch_dims + rh_contract)

    all_weak = (
        isinstance(lhs, FwdLaplArray)
        and isinstance(rhs, FwdLaplArray)
        and lhs.jacobian.weak
        and rhs.jacobian.weak
    )

    left_inp = rearrange(
        (lhs,),
        dict(
            contract_dims=lh_contract,
            batch_dims=lh_batch_dims,
            brdcast_dims=lh_brdcast_dims,
            other_brdcast_dims=rh_brdcast_dims,
        ),
        sparsity_threshold=sparsity_threshold,
    )
    right_inp = rearrange(
        (rhs,),
        dict(
            contract_dims=rh_contract,
            batch_dims=rh_batch_dims,
            brdcast_dims=rh_brdcast_dims,
            other_brdcast_dims=lh_brdcast_dims,
            rhs=True,
        ),
        sparsity_threshold=sparsity_threshold,
    )

    # ====================================================================================
    # Fast path for sparse solutions with reasonably small intermediate size
    # ====================================================================================
    # If the intermediate size is the same as the largest input size, we can also
    # accelerate things by decomposing the dot product into a sum and a multiplication.
    # This costs more memory but should generally be faster thanks to sparser operations.
    inter_size = np.prod(jnp.broadcast_shapes(left_inp.shape, right_inp.shape))
    in_size = max(left_inp.size, right_inp.size)

    def fwd_lapl_mul_sum(
        lhs: FwdLaplArray, rhs: FwdLaplArray, dims: tuple[int, ...] | int
    ):
        # Fwd Laplacian for multiply and sum
        # lazy import to avoid circular imports
        from .interpreter import forward_laplacian

        return forward_laplacian(
            lambda x, y: (x * y).sum(dims),
            disable_jit=True,
            sparsity_threshold=sparsity_threshold,
        )(lhs, rhs)

    if inter_size == in_size and all_weak:
        neg_lh_batch_dims = tuple(np.array(lh_batch_dims) - lhs.ndim)
        neg_rh_batch_dims = tuple(np.array(rh_batch_dims) - rhs.ndim)
        neg_lh_contract = tuple(np.array(lh_contract) - lhs.ndim)
        neg_rh_contract = tuple(np.array(rh_contract) - rhs.ndim)
        if (
            neg_lh_contract == neg_rh_contract
            and neg_lh_batch_dims == neg_rh_batch_dims
            and lh_brdcast_dims == tuple(range(len(lh_brdcast_dims)))
            and rh_brdcast_dims == tuple(range(len(rh_brdcast_dims)))
            and lh_batch_dims == tuple(sorted(lh_batch_dims))
            and rh_batch_dims == tuple(sorted(rh_batch_dims))
        ):
            # The checks above do the following:
            # * check that batch dims are identical in both
            # * check that contract dim are identical in both
            # * check that all brdcast dims are at the beginning
            # * check that no transpositions are needed
            # Special case where we can skip the transpositions
            return fwd_lapl_mul_sum(lhs, rhs, lh_contract)  # type: ignore
        return fwd_lapl_mul_sum(left_inp, right_inp, -1)

    # ====================================================================================
    # General solution
    # ====================================================================================
    # this einsum is somewhat inefficient.
    # one should think about rewriting the hessian
    # computation and just use the regular dot product.
    def dot_last(lhs: Array, rhs: Array) -> Array:
        return jnp.einsum(
            '...i,...i->...',
            lhs,
            rhs,
            precision=precision,
            # This flag only exists in newer JAX versions.
            preferred_element_type=preferred_element_type,
        )

    result = wrap_forward_laplacian(
        dot_last,
        flags=FunctionFlags.DOT_PRODUCT
        | FunctionFlags.JOIN_JVP
        | FunctionFlags.SPARSE_JHJ,
        in_axes=-1,
    )((left_inp, right_inp), {}, sparsity_threshold=sparsity_threshold)
    return result


def dtype_conversion(
    args: tuple[ArrayOrFwdLaplArray],
    kwargs: dict[str, Any],
    sparsity_threshold: int,
):
    return args[0].astype(kwargs['new_dtype'])


@jax.custom_jvp
def slogdet(x):
    # We only need this custom slog det to avoid a jax bug
    # https://github.com/google/jax/issues/17379
    # We explictily decompose this here as newer version will return
    # SlogDetResult which is a NamedTuple and does not combine nicely with regular tuples.
    sign, logdet = jnp.linalg.slogdet(x)
    return sign, logdet


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

    def custom_jvp(jacobian, tangent, sign):
        jac_dot_tangent = jnp.vdot(jacobian.T.conj(), tangent)
        if jac_dot_tangent.dtype in (jnp.complex64, jnp.complex128):
            # this is not the real jvp but a cached value to ease the Tr(JHJ^T) computation
            sign_jvp = jac_dot_tangent
            log_det_jvp = jac_dot_tangent.real
        else:
            sign_jvp = jnp.zeros((), dtype=jac_dot_tangent.dtype)
            log_det_jvp = jac_dot_tangent
        return (sign_jvp, log_det_jvp)

    y_tangent = jax.vmap(custom_jvp)(jacobians, tangents, sign)

    y, y_tangent = jtu.tree_map(lambda x: x.reshape(*batch_shape), (y, y_tangent))
    return y, y_tangent


slogdet.defjvp(slogdet_jvp)


def slogdet_wrapper(
    x: tuple[ArrayOrFwdLaplArray],
    kwargs: dict[str, Any],
    sparsity_threshold: int,
):
    fwd_lapl_fn = wrap_forward_laplacian(
        slogdet, custom_jac_hessian_jac=slogdet_jac_hessian_jac
    )
    sign, logdet = fwd_lapl_fn(x, {}, sparsity_threshold=0)
    # Remove the jacobian of the sign
    if jax.dtypes.issubdtype(sign.dtype, jnp.complexfloating):
        sign_jac = sign.jacobian.data
        sign_jac_flat = sign_jac.reshape(-1, *sign.shape).imag
        sign_jac_dot = jnp.einsum('i...,i...->...', sign_jac_flat, sign_jac_flat)
        sign = FwdLaplArray(
            sign.x,
            FwdJacobian(1.0j * sign.x * sign_jac.imag, x0_idx=sign.jacobian.x0_idx),
            sign.x * (1.0j * sign.laplacian.imag - sign_jac_dot),
        )
    if jax.dtypes.issubdtype(sign.dtype, jnp.floating):
        sign = sign.x
    return sign, logdet


@jax.custom_jvp
def complex_abs(x):
    return jnp.abs(x)


@complex_abs.defjvp
def complex_abs_jvp(primals, tangents):
    # This the standard JVP rule for the absolute value may cause
    # numerical issues. We use the following rule instead:
    # abs(x) = sqrt(x.real**2 + x.imag**2)
    primals, tangents = primals[0], tangents[0]
    y = complex_abs(primals)
    if not is_tree_complex(primals):
        return y, jnp.sign(primals) * tangents
    y_tangent = (primals.real * tangents.real + primals.imag * tangents.imag) / y
    return y, y_tangent


def abs_wrapper(
    x: tuple[ArrayOrFwdLaplArray],
    kwargs: dict[str, Any],
    sparsity_threshold: int,
):
    if is_tree_complex(x):
        return wrap_forward_laplacian(
            complex_abs,
            in_axes=(),
            custom_jac_hessian_jac=complex_abs_jac_hessian_jac,
        )(x, kwargs, sparsity_threshold)

    return wrap_forward_laplacian(jax.lax.abs, flags=FunctionFlags.LINEAR, in_axes=())(
        x, kwargs, sparsity_threshold
    )


_LAPLACE_FN_REGISTRY: dict[Primitive | str, ForwardLaplacian] = {
    jax.lax.conj_p: wrap_elementwise(jnp.conj),
    jax.lax.imag_p: wrap_elementwise(jnp.imag),
    jax.lax.real_p: wrap_elementwise(jnp.real),
    jax.lax.dot_general_p: dot_general,
    jax.lax.abs_p: abs_wrapper,
    jax.lax.neg_p: wrap_forward_laplacian(
        jax.lax.neg, flags=FunctionFlags.LINEAR, in_axes=()
    ),
    jax.lax.add_p: wrap_forward_laplacian(
        jax.lax.add, flags=FunctionFlags.LINEAR, in_axes=()
    ),
    jax.lax.sub_p: wrap_forward_laplacian(
        jax.lax.sub, flags=FunctionFlags.LINEAR, in_axes=()
    ),
    jax.lax.mul_p: wrap_forward_laplacian(
        jax.lax.mul, flags=FunctionFlags.MULTIPLICATION, in_axes=()
    ),
    jax.lax.div_p: wrap_forward_laplacian(
        jax.lax.div,
        flags=FunctionFlags.LINEAR_IN_FIRST,
        in_axes=(),
        custom_jac_hessian_jac=div_jac_hessian_jac,
    ),
    jax.lax.pow_p: wrap_forward_laplacian(jax.lax.pow, in_axes=()),
    jax.lax.integer_pow_p: wrap_forward_laplacian(jax.lax.integer_pow, in_axes=()),
    jax.lax.sign_p: warp_without_fwd_laplacian(jax.lax.sign),
    jax.lax.reduce_sum_p: wrap_forward_laplacian(
        jax.lax.reduce_sum_p.bind,
        flags=FunctionFlags.SUMMATION,
        name='reduce_sum',
    ),
    jax.lax.reduce_max_p: wrap_forward_laplacian(
        jax.lax.reduce_max_p.bind,
        flags=FunctionFlags.REDUCTION | FunctionFlags.LINEAR,
        name='reduce_max',
    ),
    jax.lax.reduce_min_p: wrap_forward_laplacian(
        jax.lax.reduce_min_p.bind,
        flags=FunctionFlags.REDUCTION | FunctionFlags.LINEAR,
        name='reduce_min',
    ),
    jax.lax.reduce_prod_p: wrap_forward_laplacian(
        jax.lax.reduce_prod_p.bind, flags=FunctionFlags.REDUCTION, name='reduce_prod'
    ),
    jax.lax.cumsum_p: wrap_forward_laplacian(
        jax.lax.cumsum, flags=FunctionFlags.LINEAR
    ),
    jax.lax.sqrt_p: wrap_forward_laplacian(jax.lax.sqrt, in_axes=()),
    jax.lax.rsqrt_p: wrap_forward_laplacian(jax.lax.rsqrt, in_axes=()),
    jax.lax.log_p: wrap_forward_laplacian(jax.lax.log, in_axes=()),
    jax.lax.log1p_p: wrap_forward_laplacian(jax.lax.log1p, in_axes=()),
    jax.lax.exp_p: wrap_forward_laplacian(jax.lax.exp, in_axes=()),
    jax.lax.expm1_p: wrap_forward_laplacian(jax.lax.expm1, in_axes=()),
    jax.lax.tanh_p: wrap_forward_laplacian(jax.lax.tanh, in_axes=()),
    jax.lax.logistic_p: wrap_forward_laplacian(jax.lax.logistic, in_axes=()),
    jax.lax.acos_p: wrap_forward_laplacian(jax.lax.acos, in_axes=()),
    jax.lax.asin_p: wrap_forward_laplacian(jax.lax.asin, in_axes=()),
    jax.lax.atan_p: wrap_forward_laplacian(jax.lax.atan, in_axes=()),
    jax.lax.atan2_p: wrap_forward_laplacian(jax.lax.atan2, in_axes=()),
    jax.lax.cos_p: wrap_forward_laplacian(jax.lax.cos, in_axes=()),
    jax.lax.sin_p: wrap_forward_laplacian(jax.lax.sin, in_axes=()),
    jax.lax.tan_p: wrap_forward_laplacian(jax.lax.tan, in_axes=()),
    jax.lax.broadcast_in_dim_p: wrap_forward_laplacian(
        jax.lax.broadcast_in_dim, flags=FunctionFlags.INDEXING
    ),
    jax.lax.reshape_p: wrap_forward_laplacian(
        jax.lax.reshape, flags=FunctionFlags.INDEXING
    ),
    jax.lax.slice_p: wrap_forward_laplacian(
        jax.lax.slice, flags=FunctionFlags.INDEXING
    ),
    jax.lax.dynamic_slice_p: wrap_forward_laplacian(
        jax.lax.dynamic_slice_p.bind,
        flags=FunctionFlags.INDEXING,
        name='slice',
        index_static_args=slice(1, None),
    ),
    jax.lax.concatenate_p: wrap_forward_laplacian(
        jax.lax.concatenate_p.bind,
        flags=FunctionFlags.INDEXING,
        name='concatenate',
        index_static_args=(),
    ),
    jax.lax.select_n_p: wrap_forward_laplacian(
        jax.lax.select_n, flags=FunctionFlags.INDEXING, index_static_args=(0,)
    ),
    jax.lax.gather_p: wrap_forward_laplacian(
        jax.lax.gather_p.bind, flags=FunctionFlags.INDEXING, name='gather'
    ),
    jax.lax.transpose_p: wrap_forward_laplacian(
        jax.lax.transpose, flags=FunctionFlags.INDEXING
    ),
    jax.lax.squeeze_p: wrap_forward_laplacian(
        jax.lax.squeeze, flags=FunctionFlags.INDEXING
    ),
    jax.lax.rev_p: wrap_forward_laplacian(jax.lax.rev, flags=FunctionFlags.INDEXING),
    jax.lax.max_p: wrap_forward_laplacian(
        jax.lax.max, in_axes=(), flags=FunctionFlags.LINEAR
    ),
    jax.lax.min_p: wrap_forward_laplacian(
        jax.lax.min, in_axes=(), flags=FunctionFlags.LINEAR
    ),
    jax.lax.scatter_p: wrap_forward_laplacian(
        jax.lax.scatter_p.bind,
        flags=FunctionFlags.INDEXING | FunctionFlags.SCATTER,
        name='scatter',
    ),
    # The current scatter implementation is frequently slower than the naive approach.
    # TODO: add scatter flag back in once the scatter implementation improves.
    jax.lax.scatter_add_p: wrap_forward_laplacian(
        jax.lax.scatter_add_p.bind,
        flags=FunctionFlags.LINEAR,
        name='scatter_add',
    ),
    jax.lax.stop_gradient_p: warp_without_fwd_laplacian(jax.lax.stop_gradient),
    jax.lax.eq_p: warp_without_fwd_laplacian(jax.lax.eq),
    jax.lax.lt_p: warp_without_fwd_laplacian(jax.lax.lt),
    jax.lax.le_p: warp_without_fwd_laplacian(jax.lax.le),
    jax.lax.gt_p: warp_without_fwd_laplacian(jax.lax.gt),
    jax.lax.ge_p: warp_without_fwd_laplacian(jax.lax.ge),
    jax.lax.ne_p: warp_without_fwd_laplacian(jax.lax.ne),
    jax.lax.xor_p: warp_without_fwd_laplacian(jax.lax.bitwise_xor),
    jax.lax.not_p: warp_without_fwd_laplacian(jax.lax.bitwise_not),
    jax.lax.and_p: warp_without_fwd_laplacian(jax.lax.bitwise_and),
    jax.lax.or_p: warp_without_fwd_laplacian(jax.lax.bitwise_or),
    jax.lax.is_finite_p: warp_without_fwd_laplacian(jax.lax.is_finite),
    jax.lax.convert_element_type_p: dtype_conversion,
    'sign': warp_without_fwd_laplacian(jax.lax.sign),
    'logaddexp': wrap_forward_laplacian(jnp.logaddexp, in_axes=()),
    'sigmoid': wrap_forward_laplacian(jax.nn.sigmoid, in_axes=()),
    'softplus': wrap_forward_laplacian(jax.nn.softplus, in_axes=()),
    'silu': wrap_forward_laplacian(jax.nn.silu, in_axes=()),
    'slogdet': slogdet_wrapper,
}


def register_function(primitive_or_name: Primitive | str, laplacian: ForwardLaplacian):
    """
    Register a function or primitive with a forward laplacian.
    """
    _LAPLACE_FN_REGISTRY[primitive_or_name] = laplacian


def deregister_function(primitive_or_name: Primitive | str):
    """
    Deregister a function or primitive.
    """
    del _LAPLACE_FN_REGISTRY[primitive_or_name]


def is_registered(primitive_or_name: Primitive | str) -> bool:
    """
    Check whether a primitive or function name is registered.
    """
    return primitive_or_name in _LAPLACE_FN_REGISTRY


@overload
def get_laplacian(
    primitive_or_name: Primitive, wrap_if_missing: Literal[True]
) -> ForwardLaplacian: ...


@overload
def get_laplacian(
    primitive_or_name: Primitive | str, wrap_if_missing: Literal[False] = False
) -> ForwardLaplacian | None: ...


def get_laplacian(
    primitive_or_name: Primitive | str, wrap_if_missing: bool = False
) -> ForwardLaplacian | None:
    """
    Get the forward laplacian of a primitive or a function name.
    If the function is not registered, it will return None or a default wrap if wrap_if_missing is True.

    Args:
        primitive_or_name: The primitive or function name.
        wrap_if_missing: If True, wrap the function in a forward laplacian if it s not registered.
    """
    if is_registered(primitive_or_name):
        return _LAPLACE_FN_REGISTRY[primitive_or_name]
    if wrap_if_missing:
        if isinstance(primitive_or_name, Primitive):
            logging.warning(
                f'{primitive_or_name} not in registry. The following call might be slow as we will compute the full hessian.'
            )
            return wrap_forward_laplacian(primitive_or_name.bind)
        else:
            raise TypeError(f"Can't wrap {primitive_or_name} based on function names.")
    return None
