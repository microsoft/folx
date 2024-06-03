import functools
import logging
from typing import Callable, Sequence

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import jaxlib.xla_extension
import numpy as np

from .ad import hessian, jacrev
from .api import (
    JAC_DIM,
    Array,
    Axes,
    CustomTraceJacHessianJac,
    ExtraArgs,
    ForwardFn,
    FunctionFlags,
    FwdJacobian,
    FwdLaplArgs,
    FwdLaplArray,
    MergeFn,
    PyTree,
)
from .utils import (
    add_vmap_jacobian_dim,
    array_wise_flat_wrap,
    compact_repeated_dims_except,
    flat_wrap,
    get_reduced_jacobians,
    jac_jacT,
    trace_jac_jacT,
    trace_of_product,
    vmap_sequences_and_squeeze,
)


def JHJ_via_hessian(flat_fn: Callable, flat_x: Array, grad_2d: Array):
    # We directly compute the hessian and then the trace of the product.
    flat_hessian = hessian(flat_fn)(flat_x)
    return trace_of_product(flat_hessian, grad_2d @ grad_2d.T)


def JHJ_via_trace(flat_fn: Callable, flat_x: Array, grad_2d: Array):
    # Directly copmute the trace of tr(HJJ^T)=tr(J^THJ)
    @functools.partial(jax.vmap, in_axes=-1, out_axes=-1)
    def vhvp(tangent):
        def vjp(x):
            @functools.partial(jax.vmap, in_axes=(None, -1), out_axes=-1)
            def jvp(x, tangent):
                return jax.jvp(flat_fn, (x,), (tangent,))[1]

            return jvp(x, grad_2d)

        return jax.jvp(vjp, (flat_x,), (tangent,))[1]

    return jnp.trace(vhvp(grad_2d), axis1=-2, axis2=-1)


def JHJ_via_hvp(flat_fn: Callable, flat_x: Array, grad_2d: Array):
    # Implementation where we compute HJ and then the trace via
    # the sum of hadamard product
    @functools.partial(jax.vmap, in_axes=-1, out_axes=-1)
    def hvp(tangent):
        return jax.jvp(jacrev(flat_fn), (flat_x,), (tangent,))[1]

    HJ = hvp(grad_2d)  # N x D x K
    return trace_of_product(HJ, grad_2d)


def general_jac_hessian_jac(
    fn: ForwardFn, args: FwdLaplArgs, materialize_idx: Array | None
):
    # It's conceptually easier to work with the flattened version of the
    # Hessian, since we can then use einsum to compute the trace.
    flat_fn = flat_wrap(fn, *args.x)
    flat_x = jfu.ravel_pytree(args.x)[0]
    out, unravel = jfu.ravel_pytree(fn(*args.x))
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
    jac_dim, inp_dim = grad_2d.shape
    is_complex_to_real = jnp.iscomplexobj(flat_x) and not jnp.iscomplexobj(out)

    if inp_dim > jac_dim:
        if is_complex_to_real:
            # Materializing the Hessian for a complex to real function is not supported.
            # We avoid this by only performing HvJ products.
            flat_out = JHJ_via_hvp(flat_fn, flat_x, grad_2d).real
        else:
            flat_out = JHJ_via_hessian(flat_fn, flat_x, grad_2d)
    else:
        # Here we contract the Jacobian dimensions directly without computing the full Hessian.
        # This might be more efficient if the Jacobian is large and the Hessian is small.
        flat_out = JHJ_via_trace(flat_fn, flat_x, grad_2d)
    return unravel(flat_out)


def off_diagblock_jac_hessian_jac(
    fn: ForwardFn, args: FwdLaplArgs, out_idx: Array | None
):
    # if we know that a function is linear in one arguments, it's hessian must be off diagonal
    # thus we can safe some computation by only computing the off diagonal part of the hessian.
    assert len(args) == 2, 'Off diag hessian only supports 2 args at the moment.'

    def flat_arr(x: FwdLaplArray) -> Array:
        return jfu.ravel_pytree(x.x)[0]

    flat_fn = array_wise_flat_wrap(fn, *args.x)

    def jac_lhs(lhs, rhs):
        return jax.jacrev(flat_fn, argnums=0)(lhs, rhs)

    hessian = jax.jacfwd(jac_lhs, argnums=1)(
        flat_arr(args.arrays[0]), flat_arr(args.arrays[1])
    )

    flat_out = 2 * trace_of_product(
        hessian,
        jac_jacT(args.arrays[0].jacobian, args.arrays[1].jacobian, out_idx),
    )
    unravel = jfu.ravel_pytree(fn(*args.x))[1]
    return unravel(flat_out)


def dot_product_jac_hessian_jac(
    fn: ForwardFn, args: FwdLaplArgs, shared_idx: Array | None
):
    # For a dot product we know that the hessian looks like this:
    # [0, I]
    # [I, 0]
    # where I is the identity matrix of the same shape as the input.
    assert len(args) == 2, 'Dot product only supports two args.'
    lhs, rhs = args.jacobian
    flat_out = 2 * trace_jac_jacT(lhs, rhs, shared_idx)[None]
    unravel = jfu.ravel_pytree(fn(*args.x))[1]
    return unravel(flat_out)


def remove_fill(arrs: np.ndarray, find_unique: bool = False):
    """
    Remove the fill value from an array. As the tensors might not be shaped correctly
    afterwards, we reduce all the leading dimensions by lists.

    Args:
        - arrs: array to remove fill value from
    Returns:
        - arrs: nested lists of arrays without fill value
    """
    if arrs.size == 0:
        return arrs
    if arrs[0].ndim >= 1:
        return [remove_fill(x, find_unique=find_unique) for x in arrs]
    if find_unique:
        arrs = np.unique(arrs)
    return arrs[arrs >= 0]  # type: ignore


def merge_and_populate(
    arrs: Sequence[np.ndarray],
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
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
        is_leaf=lambda x: isinstance(x, np.ndarray),
    )
    sizes = jtu.tree_map(
        lambda x: x.size, result, is_leaf=lambda x: isinstance(x, np.ndarray)
    )
    max_size = np.max(jtu.tree_leaves(sizes))
    result = jtu.tree_map(
        lambda x: np.concatenate([x, np.full(max_size - x.size, -1, dtype=x.dtype)]),
        result,
        is_leaf=lambda x: isinstance(x, np.ndarray),
    )
    return np.asarray(result, dtype=int)


def find_out_idx(lapl_args: FwdLaplArgs, in_axes, flags: FunctionFlags, threshold: int):
    if not lapl_args.any_jacobian_weak:
        return None, True
    # TODO: Rewrite this!! This is quity messy and inefficient.
    # it assumes that we're only interested in the last dimension.
    with jax.ensure_compile_time_eval():
        vmap_seq, (inp,) = vmap_sequences_and_squeeze(
            ([j.mask for j in lapl_args.jacobian],),
            (
                [
                    j
                    for j in add_vmap_jacobian_dim(
                        lapl_args, FwdLaplArgs(in_axes)
                    ).jacobian
                ],
            ),
        )
        max_size = np.max(
            [np.sum(j.unique_idx >= 0, dtype=int) for j in lapl_args.jacobian]
        )
        # This can be quite memory intensive, so we try to do it on the GPU and
        # if that fails we just use the CPU. On the CPU this takes quite some time.
        # TODO: work on a more memory efficient implementation!
        unique_fn = functools.partial(jnp.unique, size=max_size + 1, fill_value=-1)

        def idx_fn(x):
            return jtu.tree_map(unique_fn, x)

        for s in vmap_seq[::-1]:
            idx_fn = jax.vmap(idx_fn, in_axes=s)
        try:
            # This path is more memory intensive by using the GPU to find uniques but
            # potentially fails if the arrays are too large.
            # +1 because we need to accomodate the -1.
            arrs = np.asarray(idx_fn(inp), dtype=int)
        except jaxlib.xla_extension.XlaRuntimeError:
            logging.info(
                'Failed to find unique elements on GPU, falling back to CPU. This will be slow.'
            )
            with jax.default_device(jax.devices('cpu')[0]):
                arrs = np.asarray(idx_fn(inp), dtype=int)
        filtered_arrs = remove_fill(arrs, False)

    if FunctionFlags.LINEAR_IN_ONE in flags:
        # For off diagonal Hessians we only need to look at the intersection between
        # all arrays rather than their union.
        idx = merge_and_populate(filtered_arrs, np.intersect1d)  # type: ignore
    else:
        idx = merge_and_populate(filtered_arrs, np.union1d)  # type: ignore
    idx = np.moveaxis(idx, -1, JAC_DIM)

    if idx.shape[JAC_DIM] >= max_size or idx.shape[JAC_DIM] > threshold:
        return idx, True
    return idx, False


def remove_zero_entries(
    lapl_args: FwdLaplArgs,
    out_idx: np.ndarray,
    dense_out: bool,
):
    if dense_out:
        return lapl_args, out_idx, None

    mask = (out_idx != -1).any(0)
    if mask.sum() > 0.5 * mask.size:
        # this is a heuristic to avoid having unnecessary indexing overhead for
        # insufficiently sparse masks.
        return lapl_args, out_idx, None

    indices = np.where(mask)
    new_mat_idx = out_idx[(slice(None), *indices)]
    new_arrs = []
    for arg in lapl_args.arrays:
        brdcast_dims = np.where(np.array(arg.x.shape) == 1)[0]
        idx = tuple(0 if i in brdcast_dims else x for i, x in enumerate(indices))
        new_arrs.append(
            FwdLaplArray(
                x=arg.x[idx],
                jacobian=FwdJacobian(
                    data=arg.jacobian.data[(slice(None), *idx)],
                    x0_idx=arg.jacobian.x0_idx[(slice(None), *idx)]
                    if arg.jacobian.x0_idx is not None
                    else None,
                ),
                laplacian=arg.laplacian[idx],
            )
        )
    new_args = FwdLaplArgs(tuple(new_arrs))
    return new_args, new_mat_idx, mask


def vmapped_jac_hessian_jac(
    fwd: ForwardFn,
    flags: FunctionFlags,
    custom_jac_hessian_jac: CustomTraceJacHessianJac | None,
    extra_args: ExtraArgs,
    in_axes: Axes,
    extra_in_axes: Axes,
    merge: MergeFn,
    sparsity_threshold: int,
    lapl_args: FwdLaplArgs,
) -> PyTree[Array]:
    # Determine output structure
    def merged_fn(*x: Array):
        return fwd(*merge(x, extra_args))

    out = merged_fn(*lapl_args.x)
    unravel = jfu.ravel_pytree(out)[1]

    out_idx, dense_out = find_out_idx(lapl_args, in_axes, flags, sparsity_threshold)

    # If the output is dense, we can densify the input
    if dense_out and FunctionFlags.SPARSE_JHJ not in flags:
        lapl_args = lapl_args.dense
        out_idx = None

    # If the output is empty, we can return zeros
    if out_idx is not None and out_idx.shape[JAC_DIM] == 0:
        return jnp.zeros(())

    # If we do a dot product (not a hadamard product) we can check for empty hessian entries
    if FunctionFlags.DOT_PRODUCT in flags and all(len(a) == 1 for a in in_axes):
        lapl_args, out_idx, mask = remove_zero_entries(lapl_args, out_idx, dense_out)
        in_axes = jtu.tree_map(lambda _: -1, in_axes)
    else:
        mask = None

    # Broadcast and flatten all arguments
    vmap_seq, (lapl_args, extra_args) = vmap_sequences_and_squeeze(
        (lapl_args, extra_args),
        (add_vmap_jacobian_dim(lapl_args, FwdLaplArgs(in_axes)), extra_in_axes),
    )

    # Hessian computation
    def hess_transform(args: FwdLaplArgs, extra_args: ExtraArgs, out_idx):
        def merged_fn(*x):
            return fwd(*merge(x, extra_args))

        merged_fn.__name__ = fwd.__name__

        if FunctionFlags.SPARSE_JHJ not in flags:
            out_idx = None if dense_out else out_idx

        if custom_jac_hessian_jac is not None:
            result = custom_jac_hessian_jac(args, extra_args, merge, out_idx)
        elif FunctionFlags.MULTIPLICATION in flags:
            result = dot_product_jac_hessian_jac(merged_fn, args, out_idx)
        elif FunctionFlags.LINEAR_IN_ONE in flags:
            result = off_diagblock_jac_hessian_jac(merged_fn, args, out_idx)
        else:
            result = general_jac_hessian_jac(merged_fn, args, out_idx)
        return result

    # TODO: this implementation also assumes that we only reduce the last dimension.
    if out_idx is not None:
        # By compressing out_idx we can reduce the number of non-coalesced memory accesses.
        out_idx, compressed_axes = compact_repeated_dims_except(out_idx, JAC_DIM)
        out_idx_seq: list[int | None] = [1] * len(vmap_seq)
        for c in compressed_axes[::-1]:
            out_idx_seq[c - 1] = None
            out_idx = jnp.take(out_idx, 0, axis=c)
    else:
        out_idx_seq = [None] * len(vmap_seq)

    # vectorize the Tr(JHJ^T)
    for axes, oia in zip(vmap_seq[::-1], out_idx_seq[::-1]):
        hess_transform = jax.vmap(hess_transform, in_axes=(*axes, oia))

    # flatten to 1D and then unravel to the original structure
    result = hess_transform(lapl_args, extra_args, out_idx)
    if mask is not None:
        result = jnp.zeros_like(out).at[mask].set(result)  # type: ignore
    return unravel(jfu.ravel_pytree(result)[0])


def get_jacobian_hessian_jacobian_trace(
    fwd: ForwardFn,
    flags: FunctionFlags,
    custom_jac_hessian_jac: CustomTraceJacHessianJac | None,
    extra_args: ExtraArgs,
    in_axes: Axes,
    extra_in_axes: Axes,
    merge: MergeFn,
):
    def hessian_transform(args: FwdLaplArgs, sparsity_threshold: int):
        if FunctionFlags.LINEAR in flags:
            return jnp.zeros(())
        elif FunctionFlags.LINEAR_IN_ONE in flags and len(args.arrays) == 1:
            return jnp.zeros(())
        elif (
            FunctionFlags.LINEAR_IN_FIRST in flags
            and jtu.tree_leaves(merge(args.x, extra_args))[0] is args.x[0]
            and len(args.arrays) == 1
        ):
            return jnp.zeros(())
        else:
            return vmapped_jac_hessian_jac(
                fwd=fwd,
                flags=flags,
                custom_jac_hessian_jac=custom_jac_hessian_jac,
                extra_args=extra_args,
                in_axes=in_axes,
                extra_in_axes=extra_in_axes,
                merge=merge,
                sparsity_threshold=sparsity_threshold,
                lapl_args=args,
            )

    return hessian_transform
