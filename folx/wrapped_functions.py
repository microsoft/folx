import functools
import logging
from typing import Any, Literal, ParamSpec, TypeVar, overload

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

try:
    from jax.extend.core import Primitive
except ImportError:
    from jax.core import Primitive  # type: ignore[import-error]

from folx.ad import is_tree_complex

from .api import (
    JAC_DIM,
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
from .utils import mark_varying_like
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
        out_size = np.prod(x_rearranged.shape[-len(contract_dims) :], dtype=int)
        return x_rearranged.reshape(
            *x_rearranged.shape[: -len(contract_dims)], out_size
        )
    # If there are no contract dims, we need to add a dummy dimensions
    return x_rearranged[..., None]


def _dot_general_one_constant(
    lhs: ArrayOrFwdLaplArray,
    rhs: ArrayOrFwdLaplArray,
    dimension_numbers,
    precision,
    preferred_element_type,
    sparsity_threshold: int,
) -> FwdLaplArray | None:
    """Direct dot_general fast path when exactly one operand is a FwdLaplArray.

    For y = dot_general(h, W) with W constant, the op is linear in h, so we
    can apply jax.lax.dot_general directly to (x, jacobian.data, laplacian)
    instead of decomposing into broadcast+mul+reduce_sum through the einsum-
    on-rearranged-inputs path. ``vmap`` over JAC_DIM lowers to one higher-dim
    dot_general kernel (no Python loop).

    Sparse Jacobian: the output's per-position dependency is the input's
    x0_idx with the contract axes removed. When x0_idx varies along the
    contract axes (e.g. ``concat([diff, dist]) @ W``), the rows are first
    aligned to the per-position union mask over those axes. Returns None if
    that union exceeds the sparsity threshold.
    """
    h_is_lhs = isinstance(lhs, FwdLaplArray)
    h: FwdLaplArray = lhs if h_is_lhs else rhs  # type: ignore[assignment]
    W: Array = rhs if h_is_lhs else lhs  # type: ignore[assignment]
    (lh_contract, rh_contract), (lh_batch, rh_batch) = dimension_numbers
    h_contract = tuple(lh_contract if h_is_lhs else rh_contract)
    h_batch = tuple(lh_batch if h_is_lhs else rh_batch)
    W_contract = tuple(rh_contract if h_is_lhs else lh_contract)
    W_batch = tuple(rh_batch if h_is_lhs else lh_batch)

    def op(arr_h):
        a, b = (arr_h, W) if h_is_lhs else (W, arr_h)
        return jax.lax.dot_general(
            a,
            b,
            dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    jac_data = h.jacobian.data
    if h.jacobian.weak:
        x0_idx = h.jacobian.x0_idx
        assert x0_idx is not None
        # x0_idx has axis 0 = JAC_DIM, so h's contract dims shift by 1 on data.
        data_contract = tuple(c + 1 for c in h_contract)
        varying = any(
            not np.array_equal(
                x0_idx, np.broadcast_to(np.take(x0_idx, [0], axis=ax), x0_idx.shape)
            )
            for ax in data_contract
        )
        if varying:
            # Align rows to the per-position union mask over the contract axes.
            from .utils import get_jacobian_for_reduction

            (jac_data,), proj = get_jacobian_for_reduction((h.jacobian,), h_contract)
            if proj.shape[0] > sparsity_threshold:
                return None
        else:
            proj = x0_idx
            for ax in sorted(data_contract, reverse=True):
                proj = np.take(proj, 0, axis=ax)
        # proj shape now: (k, *h's non-contract axes in h's original order).
        h_non_contract = [i for i in range(h.x.ndim) if i not in h_contract]
        h_brdcast = [i for i in h_non_contract if i not in h_batch]
        # dot_general output ordering: batch_dims, then lhs_brdcast, then
        # rhs_brdcast. Permute proj so h's surviving axes follow that order.
        target_h_axes = list(h_batch) + h_brdcast
        ax_in_proj = {ax: 1 + h_non_contract.index(ax) for ax in h_non_contract}
        perm = [0] + [ax_in_proj[ax] for ax in target_h_axes]
        proj = np.transpose(proj, perm)
        # Insert size-1 axes for W's brdcast dims; final broadcast happens after
        # y_x materialization so we know the exact output shape.
        W_non_contract = [i for i in range(W.ndim) if i not in W_contract]
        n_W_brdcast = len([i for i in W_non_contract if i not in W_batch])
        n_batch = len(h_batch)
        if h_is_lhs:
            proj = proj.reshape(proj.shape + (1,) * n_W_brdcast)
        else:
            insert = 1 + n_batch
            proj = proj.reshape(
                proj.shape[:insert] + (1,) * n_W_brdcast + proj.shape[insert:]
            )
    else:
        proj = None

    h_x, h_lapl = h.x, h.laplacian
    if jac_data.shape[JAC_DIM] >= 128:
        # Keep XLA's dot merger from concatenating the small x/laplacian dots
        # into the large Jacobian dot: the resulting concatenate-rooted fusion
        # has much worse memory throughput than a plain elementwise fusion.
        # Only worth it for wide Jacobians; below the threshold the merged
        # dot is neutral-to-better.
        h_x, h_lapl = jax.lax.optimization_barrier((h_x, h_lapl))
    y_x = op(h_x)
    y_data = jax.vmap(op, in_axes=0, out_axes=0)(jac_data)
    y_lapl = op(h_lapl)
    new_x0_idx = None if proj is None else np.broadcast_to(proj, y_data.shape)
    return FwdLaplArray(y_x, FwdJacobian(y_data, new_x0_idx), y_lapl)


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

    # Fast path: exactly one operand has a Jacobian (the common h @ W case).
    # f is linear in the FwdLaplArray operand so we can just dot_general through
    # x, jacobian.data, and laplacian — skipping the rearrange + dot_last
    # decomposition that the general path uses.
    if isinstance(lhs, FwdLaplArray) ^ isinstance(rhs, FwdLaplArray):
        fast = _dot_general_one_constant(
            lhs,
            rhs,
            dimension_numbers,
            precision,
            preferred_element_type,
            sparsity_threshold,
        )
        if fast is not None:
            return fast

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
    # Decomposing the dot product into a multiplication and a sum materializes the
    # product's sparse Jacobian of size k_mul * inter_size (k_mul = summed sparse rows
    # of both operands). The general path materializes dense Jacobians of size
    # n_dense * in_size instead. Use the decomposition whenever it is the cheaper one;
    # it should generally be faster thanks to sparser operations.
    inter_size = np.prod(jnp.broadcast_shapes(left_inp.shape, right_inp.shape))
    in_size = max(left_inp.size, right_inp.size)
    use_mul_sum = False
    if all_weak:
        assert isinstance(lhs, FwdLaplArray) and isinstance(rhs, FwdLaplArray)
        lh_mask, rh_mask = lhs.jacobian.x0_idx, rhs.jacobian.x0_idx
        assert lh_mask is not None and rh_mask is not None
        if lh_mask.size > 0 and rh_mask.size > 0:
            k_mul = lhs.jacobian.data.shape[JAC_DIM] + rhs.jacobian.data.shape[JAC_DIM]
            n_dense = max(lhs.jacobian.max_n, rhs.jacobian.max_n) + 1
            use_mul_sum = bool(
                inter_size == in_size or k_mul * inter_size < n_dense * in_size
            )

    if use_mul_sum:
        # The rearranged inputs reproduce dot_general exactly via broadcasting
        # (batch, lhs_brdcast, rhs_brdcast, contract), so this is valid for any
        # dimension numbers.
        def mul_sum(x, y):
            # dot_general accumulates in preferred_element_type; cast so the
            # decomposition matches its output dtype and precision.
            if preferred_element_type is not None:
                x = x.astype(preferred_element_type)
                y = y.astype(preferred_element_type)
            return (x * y).sum(-1)

        # lazy import to avoid circular imports
        from .interpreter import forward_laplacian

        return forward_laplacian(
            mul_sum,
            disable_jit=True,
            sparsity_threshold=sparsity_threshold,
        )(left_inp, right_inp)

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
            sign_jvp = mark_varying_like(
                jnp.zeros((), dtype=jac_dot_tangent.dtype), jac_dot_tangent
            )
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
        jax.lax.broadcast_in_dim_p.bind,
        flags=FunctionFlags.INDEXING,
        name='broadcast_in_dim',
    ),
    jax.lax.reshape_p: wrap_forward_laplacian(
        jax.lax.reshape_p.bind, flags=FunctionFlags.INDEXING, name='reshape'
    ),
    jax.lax.slice_p: wrap_forward_laplacian(
        jax.lax.slice_p.bind, flags=FunctionFlags.INDEXING, name='slice'
    ),
    jax.lax.dynamic_slice_p: wrap_forward_laplacian(
        jax.lax.dynamic_slice_p.bind,
        flags=FunctionFlags.INDEXING,
        name='dynamic_slice',
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
        jax.lax.transpose_p.bind, flags=FunctionFlags.INDEXING, name='transpose'
    ),
    jax.lax.squeeze_p: wrap_forward_laplacian(
        jax.lax.squeeze_p.bind, flags=FunctionFlags.INDEXING, name='squeeze'
    ),
    jax.lax.rev_p: wrap_forward_laplacian(
        jax.lax.rev_p.bind, flags=FunctionFlags.INDEXING, name='rev'
    ),
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
    jax.lax.scatter_max_p: wrap_forward_laplacian(
        jax.lax.scatter_max_p.bind,
        flags=FunctionFlags.LINEAR,
        name='scatter_max',
    ),
    jax.lax.scatter_min_p: wrap_forward_laplacian(
        jax.lax.scatter_min_p.bind,
        flags=FunctionFlags.LINEAR,
        name='scatter_min',
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


# Only supported in newer JAX versions
if hasattr(jax.lax, 'square_p'):
    register_function(
        jax.lax.square_p, wrap_forward_laplacian(jax.lax.square, in_axes=())
    )
if hasattr(jax.lax, 'split_p'):
    register_function(
        jax.lax.split_p,
        wrap_forward_laplacian(
            jax.lax.split,
            flags=FunctionFlags.INDEXING,
            index_static_args=(1, 2),
        ),
    )
