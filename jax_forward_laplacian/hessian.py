import functools
import logging

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import jaxlib.xla_extension
import numpy as np
from jax import core

from .api import (
    JAC_DIM,
    Array,
    Axes,
    ExtraArgs,
    ForwardFn,
    FunctionFlags,
    FwdLaplArgs,
    FwdLaplArray,
    MergeFn,
)
from .utils import (
    add_vmap_jacobian_dim,
    array_wise_flat_wrap,
    flat_wrap,
    get_reduced_jacobians,
    jac_jacT,
    trace_jac_jacT,
    trace_of_product,
    vmap_sequences_and_squeeze,
)


def general_jac_hessian_jac(fn: ForwardFn, args: FwdLaplArgs, materialize_idx: Array | None):
    # It's conceptually easier to work with the flattened version of the
    # Hessian, since we can then use einsum to compute the trace.
    flat_fn = flat_wrap(fn, *args.x)
    flat_x = jfu.ravel_pytree(args.x)[0]
    # We have to decide on an order in which we execute tr(HJJ^T).
    # H will be of shape NxDxD, J is DxK where N could potentially be D.
    # We will do the following:
    # if K >= D, we compute
    # JJ^T first and then the trace.
    # if D < K, we compute HJ first and then the trace.
    # We should also flatten our gradient tensor to a 2D matrix where the first dimension
    # is the x0 dim and the second dim is the input dim.
    grads_2d = get_reduced_jacobians(*args.jacobian, idx=materialize_idx)
    grad_2d = jnp.concatenate([x.T for x in grads_2d], axis=0)
    D, K = grad_2d.shape
    if K > D:
        # jax.hessian uses Fwd on Reverse AD
        flat_hessian = jax.hessian(flat_fn)(flat_x)
        flat_out = trace_of_product(flat_hessian, grad_2d @ grad_2d.T)
    elif D > K:
        # Directly copmute the trace of tr(HJJ^T)=tr(J^THJ)
        @functools.partial(jax.vmap, in_axes=-1, out_axes=-1)
        def vhvp(tangent):
            def vjp(x):
                @functools.partial(jax.vmap, in_axes=(None, -1), out_axes=-1)
                def jvp(x, tangent):
                    return jax.jvp(flat_fn, (x,), (tangent,))[1]

                return jvp(x, grad_2d)

            return jax.jvp(vjp, (flat_x,), (tangent,))[1]

        flat_out = jnp.trace(vhvp(grad_2d), axis1=-2, axis2=-1)
    else:
        # Implementation where we compute HJ and then the trace via
        # the sum of hadamard product
        @functools.partial(jax.vmap, in_axes=-1, out_axes=-1)
        def hvp(tangent):
            def jacobian(x):
                return jax.jacrev(flat_fn)(x)

            return jax.jvp(jacobian, (flat_x,), (tangent,))[1]

        HJ = hvp(grad_2d)  # N x D x K
        flat_out = trace_of_product(HJ, grad_2d)  # N x D x K and D x K

    # since f(x) and nabla f(x) should have the same structure, we can use the
    # structure of f(x) to unravel the flat_out
    unravel = jfu.ravel_pytree(fn(*args.x))[1]
    return unravel(flat_out)


def off_diag_jac_hessian_jac(fn: ForwardFn, args: FwdLaplArgs, materialize_idx: Array | None):
    # if we know that a function is linear in one arguments, it's hessian must be off diagonal
    # thus we can safe some computation by only computing the off diagonal part of the hessian.
    assert len(args) == 2, "Off diag hessian only supports 2 args at the moment."

    def flat_arr(x: FwdLaplArray) -> Array:
        return jfu.ravel_pytree(x.x)[0]

    flat_fn = array_wise_flat_wrap(fn, *args.x)

    def jac_lhs(lhs, rhs):
        return jax.jacobian(flat_fn, argnums=0)(lhs, rhs)

    hessian = jax.jacobian(jac_lhs, argnums=1)(flat_arr(args.arrays[0]), flat_arr(args.arrays[1]))

    flat_out = 2 * trace_of_product(
        hessian, jac_jacT(args.arrays[0].jacobian, args.arrays[1].jacobian, materialize_idx)
    )
    unravel = jfu.ravel_pytree(fn(*args.x))[1]
    return unravel(flat_out)


def dot_product_jac_hessian_jac(fn: ForwardFn, args: FwdLaplArgs, shared_idx: Array | None):
    # For a dot product we know that the hessian looks like this:
    # [0, I]
    # [I, 0]
    # where I is the identity matrix of the same shape as the input.
    assert len(args) == 2, "Dot product only supports two args."
    flat_out = (
        2 * trace_jac_jacT(args.arrays[0].jacobian, args.arrays[1].jacobian, shared_idx)[None]
    )
    unravel = jfu.ravel_pytree(fn(*args.x))[1]
    return unravel(flat_out)


def slogdet_jac_hessian_jac(args: FwdLaplArgs, materialize_idx: Array | None):
    # For slogdet we know how to compute the determinant faster.
    # We can use the fact that the jacobian of logdet is A^-1.
    # Thus, the hessian is A^-1 (x) A^-T. Where (x) is the kronecker product.
    # We can now reformulate this to (A^-1 (x) I)(A^-1 (x) I)^T.
    # If one wants to compute the product vec(M)(A^-1 (x) I), this can be
    # efficiently evaluated as vec(MA^-1). As we multiply the Hessian from
    # both sides with the jacobian tr(JHJ^T), this can be efficiently be done
    # as tr(J@A^-1 @ A^-1^T@J^T) where the inner @ is the outer product.
    assert len(args.x) == 1
    A = args.x[0]
    A_inv = jnp.linalg.inv(A)
    J = args.jacobian[0].construct_jac_for(materialize_idx)
    J = jnp.moveaxis(J, JAC_DIM, -1)
    leading_dims = A.shape[:-2]
    x0_dim = J.shape[-1]

    def elementwise(A_inv, J):
        # Naive implementation
        # @functools.partial(jax.vmap, in_axes=(-1, None), out_axes=-1)
        # @functools.partial(jax.vmap, in_axes=(None, -1), out_axes=-1)
        # def inner(v1, v2):
        #     A_inv_v = A_inv@v1
        #     v_A_inv = v2.T@A_inv.T
        #     return -v_A_inv.reshape(-1)@A_inv_v.reshape(-1)
        # vHv = inner(J, J)
        # trace = jnp.trace(vHv)

        # We can do better and compute the trace more efficiently.
        A_inv_J = jnp.einsum("ij,jdk->idk", A_inv, J)
        trace = -trace_of_product(
            jnp.transpose(A_inv_J, (1, 0, 2)).reshape(-1, x0_dim), A_inv_J.reshape(-1, x0_dim)
        )
        return jnp.zeros(()), trace

    A_inv = A_inv.reshape(-1, *A.shape[-2:])
    J = J.reshape(-1, *J.shape[-3:])

    # We can either use vmap or scan. Scan is slightly slower but uses less memory.
    # Here we assume that we will in general encounter larger determinants rather than many.
    # signs, flat_out = jax.vmap(elementwise)(A_inv, J)
    def scan_wrapper(_, x):
        return None, elementwise(*x)

    signs, flat_out = jax.lax.scan(scan_wrapper, None, (A_inv, J))[1]
    return signs.reshape(leading_dims), flat_out.reshape(leading_dims)


def remove_fill(arrs: np.ndarray, find_unique: bool = False):
    """
    Remove the fill value from an array. As the tensors might not be shaped correctly
    afterwards, we reduce all the leading dimensions by lists.

    Args:
        - arrs: array to remove fill value from
    Returns:
        - arrs: nested lists of arrays without fill value
    """
    if arrs.ndim > 1:
        return [remove_fill(x) for x in arrs]
    if find_unique:
        arrs = np.unique(arrs)
    return arrs[arrs >= 0]  # type: ignore


def merge_and_populate(arrs, operation):
    """
    The arrays are assumed to be of the same shape. We look at the intersection of all arrays.
    We then find the maximum intersection size and fill all arrays to that size.

    Args:
        - arrs: list of arrays
    Returns:
        - arrs: np.ndarray where only intersections are kept and all arrays are filled to the same size.
    """
    result = jtu.tree_map(
        lambda *x: functools.reduce(operation, tuple(x[1:]), x[0]),
        *arrs,
        is_leaf=lambda x: isinstance(x, np.ndarray)
    )
    sizes = jtu.tree_map(lambda x: x.size, result, is_leaf=lambda x: isinstance(x, np.ndarray))
    max_size = np.max(jtu.tree_leaves(sizes))
    result = jtu.tree_map(
        lambda x: np.concatenate([x, np.full(max_size - x.size, -1, dtype=x.dtype)]),
        result,
        is_leaf=lambda x: isinstance(x, np.ndarray),
    )
    return np.asarray(result, dtype=int)


def find_materialization_idx(lapl_args: FwdLaplArgs, in_axes, flags: FunctionFlags, threshold: int):
    if not lapl_args.any_jacobian_weak:
        return None
    # TODO: Rewrite this!! This is quity messy and inefficient.
    # it assumes that we're only interested in the last dimension.
    with core.new_main(core.EvalTrace, dynamic=True):
        vmap_seq, (inp,) = vmap_sequences_and_squeeze(
            ([j.mask for j in lapl_args.jacobian],),
            ([j for j in add_vmap_jacobian_dim(lapl_args, FwdLaplArgs(in_axes)).jacobian],),
        )
        max_size = np.max([np.sum(j.unique_idx >= 0, dtype=int) for j in lapl_args.jacobian])
        # This can be quite memory intensive, so we try to do it on the GPU and
        # if that fails we just use the CPU. On the CPU this takes quite some time.
        # TODO: work on a more memory efficient implementation!
        try:
            # This path is more memory intensive by using the GPU to find uniques but
            # potentially fails if the arrays are too large.
            # +1 because we need to accommodate the -1.
            unique_fn = functools.partial(jnp.unique, size=max_size + 1, fill_value=-1)

            def idx_fn(x):
                return jtu.tree_map(unique_fn, x)

            for s in vmap_seq[::-1]:
                idx_fn = jax.vmap(idx_fn, in_axes=s)
            arrs = np.asarray(idx_fn(inp), dtype=int)
            filtered_arrs = remove_fill(arrs, False)
        except jaxlib.xla_extension.XlaRuntimeError:
            logging.info(
                "Failed to find unique elements on GPU, falling back to CPU. This will be slow."
            )
            # materialize the full array on CPU and then use numpy to sequentially find
            # unique elements.
            def identity(x):
                return x

            for s in vmap_seq[::-1]:
                identity = jax.vmap(identity, in_axes=s)
            with jax.default_device(jax.devices("cpu")[0]):
                arrs = np.asarray(identity(inp), dtype=int)
            filtered_arrs = remove_fill(arrs, True)

    if FunctionFlags.LINEAR_IN_ONE in flags:
        # For off diagonal Hessians we only need to look at the intersection between
        # all arrays rather than their union.
        idx = merge_and_populate(filtered_arrs, np.intersect1d)
    else:
        idx = merge_and_populate(filtered_arrs, np.union1d)
    idx = np.moveaxis(idx, -1, JAC_DIM)

    if idx.shape[JAC_DIM] >= max_size or idx.shape[JAC_DIM] > threshold:
        idx = None
    return idx


def vmapped_jac_hessian_jac(
    unmerged_fn: ForwardFn,
    lapl_args: FwdLaplArgs,
    in_axes,
    extra_args: ExtraArgs,
    extra_in_axes,
    merge: MergeFn,
    flags: FunctionFlags,
    sparsity_threshold: int,
):
    # Determine output structure
    def merged_fn(*x: Array):
        return unmerged_fn(*merge(x, extra_args))

    unravel = jfu.ravel_pytree(merged_fn(*lapl_args.x))[1]

    materialize_idx = find_materialization_idx(lapl_args, in_axes, flags, sparsity_threshold)
    if materialize_idx is None:
        lapl_args = lapl_args.dense
    if materialize_idx is not None and materialize_idx.shape[JAC_DIM] == 0:
        return jnp.zeros(())

    # Broadcast and flatten all arguments
    vmap_seq, (lapl_args, extra_args) = vmap_sequences_and_squeeze(
        (lapl_args, extra_args),
        (add_vmap_jacobian_dim(lapl_args, FwdLaplArgs(in_axes)), extra_in_axes),
    )
    # Hessian computation
    def hess_transform(args: FwdLaplArgs, extra_args: ExtraArgs, materialize_idx):
        def merged_fn(*x):
            return unmerged_fn(*merge(x, extra_args))

        if FunctionFlags.DOT_PRODUCT in flags:
            result = dot_product_jac_hessian_jac(merged_fn, args, materialize_idx)
        elif FunctionFlags.LINEAR_IN_ONE in flags:
            result = off_diag_jac_hessian_jac(merged_fn, args, materialize_idx)
        elif FunctionFlags.SLOGDET in flags:
            result = slogdet_jac_hessian_jac(args, materialize_idx)
        else:
            result = general_jac_hessian_jac(merged_fn, args, materialize_idx)
        return result

    # TODO: this implementation also assumes that we only reduce the last dimension.
    for axes in vmap_seq[::-1]:
        hess_transform = jax.vmap(
            hess_transform, in_axes=(*axes, (None if materialize_idx is None else 1))
        )
    # flatten to 1D and then unravel to the original structure
    flat_out = jfu.ravel_pytree(hess_transform(lapl_args, extra_args, materialize_idx))[0]
    return unravel(flat_out)


def get_jacobian_hessian_jacobian_trace(
    fwd: ForwardFn,
    fn_flags: FunctionFlags,
    extra_args: ExtraArgs,
    in_axes: Axes,
    extra_in_axes: Axes,
    merge: MergeFn,
):
    def hessian_transform(args: FwdLaplArgs, sparsity_threshold: int):
        if FunctionFlags.LINEAR in fn_flags:
            return jnp.zeros(())
        elif FunctionFlags.LINEAR_IN_ONE in fn_flags and len(args.arrays) == 1:
            return jnp.zeros(())
        elif (
            FunctionFlags.LINEAR_IN_FIRST in fn_flags
            and jtu.tree_leaves(merge(args.x, extra_args))[0] is args.x[0]
            and len(args.arrays) == 1
        ):
            return jnp.zeros(())
        else:
            return vmapped_jac_hessian_jac(
                fwd, args, in_axes, extra_args, extra_in_axes, merge, fn_flags, sparsity_threshold
            )

    return hessian_transform
