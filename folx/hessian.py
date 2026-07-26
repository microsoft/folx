import functools
from typing import Callable, Sequence

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
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


def _has_duplicate_x0_idx(x0_idx) -> bool:
    """True iff any output position has duplicate valid (>=0) entries along JAC_DIM.

    Sparse Jacobians can pick up duplicate masks from FwdJacobian addition
    (concatenation of unequal masks). Duplicates would break the fast path's
    sum-of-squares identity, so it has to fall back when they're present.
    Operates on numpy compile-time masks so it's free at runtime.
    """
    if x0_idx is None or x0_idx.shape[JAC_DIM] <= 1:
        return False
    sorted_idx = np.sort(x0_idx, axis=JAC_DIM)
    a = np.take(sorted_idx, np.arange(sorted_idx.shape[JAC_DIM] - 1), axis=JAC_DIM)
    b = np.take(sorted_idx, np.arange(1, sorted_idx.shape[JAC_DIM]), axis=JAC_DIM)
    return bool(((a == b) & (a >= 0)).any())


def elementwise_jhj_trace(fn: ForwardFn, args: FwdLaplArgs):
    """Direct tr(J^T H J) for scalar-to-scalar elementwise nonlinearities.

    For y = f(x) applied elementwise, the input-space Hessian H_y_x = f''(x) is
    diagonal, so tr(J^T H J)[i] = f''(x[i]) * sum_k J[k, i]^2. Works for both
    sparse and dense Jacobians: sum_k runs over sparse rows or dense input
    rows respectively, and both equal |∇y[i]|^2 -- provided the sparse x0_idx
    has no duplicates along JAC_DIM (caller checks).

    Avoids find_out_idx (the output mask equals the input mask for elementwise),
    the per-output vmap stacking in vmapped_jac_hessian_jac, and the nested
    JVP-of-VJP inside JHJ_via_trace. f''(x) is computed via forward-mode JVP
    of JVP (cheaper than VJP-of-JVP for scalar elementwise functions: avoids
    materializing a reverse-mode cotangent intermediate).
    """
    x = args.x[0]
    J = args.arrays[0].jacobian.data
    ones = jnp.ones_like(x)

    def df(x):
        # f'(x) via forward-mode JVP at tangent=ones; for scalar-elementwise
        # f, jvp returns f'(x) * 1 = f'(x).
        return jax.jvp(fn, (x,), (ones,))[1]

    _, second_grad = jax.jvp(df, (x,), (ones,))
    return second_grad * (J * J).sum(axis=JAC_DIM)


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


def _vmap_axes_to_original(axes_for_seq: Sequence[int | None]) -> list[int | None]:
    """Translate iteratively-reduced vmap axis indices to original-array indices.

    `axes_for_seq[j]` is the axis to vmap over at the j-th vmap, in the array
    *after* the previous (j-1) vmaps have peeled their axes off. This converts
    each entry back to an axis index in the original (un-reduced) array.
    """
    original_axes: list[int | None] = []
    removed: list[int] = []
    for ax_red in axes_for_seq:
        if ax_red is None:
            original_axes.append(None)
            continue
        ax_orig = ax_red
        for r in sorted(removed):
            if ax_orig >= r:
                ax_orig += 1
        original_axes.append(ax_orig)
        removed.append(ax_orig)
    return original_axes


def _align_mask_for_broadcast(
    mask: np.ndarray, axes_for_seq: Sequence[int | None], num_vmap_dims: int
) -> np.ndarray:
    """Reshape a mask to `(*vmap_dim_sizes, K_flat)` for per-position set ops.

    Vmap (broadcast) axes are moved to the front in `axes_for_seq` order, with
    a size-1 axis inserted wherever an entry is None. The remaining (kept) axes
    are flattened into a single trailing K dim.
    """
    original_axes = _vmap_axes_to_original(axes_for_seq)
    real_axes = [a for a in original_axes if a is not None]
    other_axes = [a for a in range(mask.ndim) if a not in real_axes]
    permuted = np.transpose(mask, real_axes + other_axes)
    for j, orig in enumerate(original_axes):
        if orig is None:
            permuted = np.expand_dims(permuted, axis=j)
    s_shape = permuted.shape[:num_vmap_dims]
    k_flat = int(np.prod(permuted.shape[num_vmap_dims:], dtype=int))
    return permuted.reshape((*s_shape, k_flat))


def _per_position_sorted_unique(arr: np.ndarray) -> np.ndarray:
    """Sorted unique non-negative values along the last axis, padded with -1.

    Args:
        arr: shape `(*S, K)`, entries are indices (`>= 0`) or `-1` (fill).
    Returns:
        Array of shape `(*S, M)` where `M` is the maximum per-position count of
        unique non-negative values, sorted ascending, padded with `-1`.
    """
    leading = arr.shape[:-1]
    if arr.shape[-1] == 0:
        return np.full((*leading, 0), -1, dtype=arr.dtype)
    sorted_arr = np.sort(arr, axis=-1)
    prev = np.concatenate(
        [np.full((*leading, 1), -2, dtype=arr.dtype), sorted_arr[..., :-1]],
        axis=-1,
    )
    is_first = (sorted_arr != prev) & (sorted_arr >= 0)
    sentinel = np.iinfo(arr.dtype).max
    masked = np.where(is_first, sorted_arr, sentinel)
    final = np.sort(masked, axis=-1)
    counts = is_first.sum(axis=-1)
    max_count = int(counts.max()) if counts.size > 0 else 0
    result = final[..., :max_count]
    return np.where(result == sentinel, -1, result)


def _per_position_intersection(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Sorted intersection of input sets along the last axis, padded with -1.

    Each input contributes one set per position; the intersection is the values
    present in every input's set. After deduplicating each input first, a value
    is in the intersection iff it occupies a length-N run in the sorted union.
    """
    n = len(masks)
    per_input = [_per_position_sorted_unique(m) for m in masks]
    combined = np.concatenate(per_input, axis=-1)
    leading = combined.shape[:-1]
    total = combined.shape[-1]
    if total < n:
        return np.full((*leading, 0), -1, dtype=combined.dtype)
    sorted_combined = np.sort(combined, axis=-1)
    first = sorted_combined[..., : total - n + 1]
    last = sorted_combined[..., n - 1 :]
    is_intersection = (first == last) & (first >= 0)
    sentinel = np.iinfo(combined.dtype).max
    masked = np.where(is_intersection, first, sentinel)
    final = np.sort(masked, axis=-1)
    counts = is_intersection.sum(axis=-1)
    max_count = int(counts.max()) if counts.size > 0 else 0
    result = final[..., :max_count]
    return np.where(result == sentinel, -1, result)


def find_out_idx(lapl_args: FwdLaplArgs, in_axes, flags: FunctionFlags, threshold: int):
    """Determine the per-output-position input dependency set for a sparse op.

    Returns `(idx, dense_out)` where `idx` has shape `(M, *broadcast_shape)` and
    `idx[:, p]` lists the sorted unique input indices that any output at
    position `p` depends on (union for general ops, intersection when only one
    arg actually couples through the Hessian). `dense_out=True` signals that
    the sparse representation isn't worth keeping at this point.

    Replaces an earlier JAX vmap-of-jnp.unique pipeline with pure NumPy: the
    masks are compile-time, so no tracing/JIT is needed and the answer is the
    same.
    """
    if not lapl_args.any_jacobian_weak:
        return None, True

    with jax.ensure_compile_time_eval():
        vmap_seq, (squeezed_masks,) = vmap_sequences_and_squeeze(
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
        squeezed_masks = [np.asarray(m) for m in squeezed_masks]

    max_size = int(
        np.max([np.sum(j.unique_idx >= 0, dtype=int) for j in lapl_args.jacobian])
    )

    num_vmap_dims = len(vmap_seq)
    # vmap_seq mirrors the input pytree structure: each entry is ([axis_per_mask],).
    aligned = [
        _align_mask_for_broadcast(m, [seq[0][i] for seq in vmap_seq], num_vmap_dims)
        for i, m in enumerate(squeezed_masks)
    ]
    s_vmap = np.broadcast_shapes(*(a.shape[:num_vmap_dims] for a in aligned))
    broadcasted = [np.broadcast_to(a, (*s_vmap, a.shape[-1])) for a in aligned]

    if FunctionFlags.LINEAR_IN_ONE in flags:
        idx = _per_position_intersection(broadcasted)
    else:
        idx = _per_position_sorted_unique(np.concatenate(broadcasted, axis=-1))

    idx = np.moveaxis(idx, -1, JAC_DIM).astype(int)

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


def _materialize_args_for_idx(lapl_args: FwdLaplArgs, out_idx: np.ndarray, in_axes):
    """Aligns all Jacobians to the static index frame ``out_idx``.

    Returns new args whose Jacobians are dense over the ``out_idx`` rows so
    the JHJ computation needs no per-position index matching. ``out_idx`` has
    the vmapped position dims only; each argument's reduced (``in_axes``)
    dims are re-inserted as size-1 axes before broadcasting.
    """
    from .utils import broadcast_except

    n_rows = out_idx.shape[JAC_DIM]
    # out_idx comes on the squeezed position frame; recover the full frame
    # (the broadcast of all args' shapes without their reduced axes).
    reduced_shapes = [
        tuple(np.delete(np.array(arr.x.shape), [a % arr.x.ndim for a in axes]))
        for arr, axes in zip(lapl_args.arrays, in_axes)
    ]
    frame = np.broadcast_shapes(*reduced_shapes)
    out_idx = out_idx.reshape(n_rows, *frame)

    new_arrays = []
    for arr, axes in zip(lapl_args.arrays, in_axes):
        pos = {a % arr.x.ndim for a in axes}
        n_lead = len(frame) - (arr.x.ndim - len(pos))
        lead, core = frame[:n_lead], iter(frame[n_lead:])
        arg_frame = [1 if a in pos else next(core) for a in range(arr.x.ndim)]
        idx = out_idx.reshape(n_rows, *lead, *arg_frame)
        jac = arr.jacobian
        if jac.weak:
            data = jac.materialize_for_idx(jac.get_index_mask(idx), n_rows)
        else:
            # Dense rows are indexed directly by x0 index.
            idx, data = broadcast_except((idx, jac.data), axis=JAC_DIM)
            take = jnp.asarray(np.maximum(idx, 0))
            data = jnp.where(
                jnp.asarray(idx >= 0),
                jnp.take_along_axis(data, take, axis=JAC_DIM),
                0,
            )
        # Materialization may expand broadcast dims; keep x/laplacian in sync
        # for the downstream vmap bookkeeping.
        pos_shape = data.shape[1:]
        new_arrays.append(
            FwdLaplArray(
                jnp.broadcast_to(arr.x, pos_shape),
                FwdJacobian.from_dense(data),
                jnp.broadcast_to(arr.laplacian, pos_shape),
            )
        )
    return FwdLaplArgs(tuple(new_arrays))


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

    # Fast path: scalar-to-scalar elementwise nonlinear ops (silu, tanh, exp, ...).
    # The Hessian is diagonal so tr(J^T H J) = f''(x) * (J*J).sum(JAC_DIM); this
    # holds for both sparse and dense Jacobian storage (the sum runs over the
    # sparse-row axis or the dense-input axis respectively, both equal to the
    # sum of squares of |∇y[i]|). Skips find_out_idx, the per-element vmap stack,
    # and the full Hessian materialization that JHJ_via_hessian would otherwise
    # do for inp_dim > jac_dim. Only valid for ops registered as elementwise
    # (empty in_axes): a matching output shape alone does not imply a diagonal
    # Hessian (e.g. cumprod). Bail if a sparse mask has duplicates --
    # sum_l J_full[l, i]^2 = sum_k J[k, i]^2 requires unique x0_idx values per
    # output position.
    if (
        custom_jac_hessian_jac is None
        and int(flags) == int(FunctionFlags.GENERAL)
        and len(lapl_args.arrays) == 1
        and all(ax == () for ax in in_axes)
        and isinstance(out, Array)
        and not _has_duplicate_x0_idx(lapl_args.arrays[0].jacobian.x0_idx)
    ):
        x0 = lapl_args.x[0]
        if out.shape == x0.shape and out.dtype == x0.dtype:
            return elementwise_jhj_trace(merged_fn, lapl_args)

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

    # For elementwise ops the index frame is static and per-position, so align
    # all Jacobians to it once via static gathers here rather than matching
    # masks at runtime inside the vmap. For dot-style ops (SPARSE_JHJ) the
    # operands broadcast against each other and pre-materializing would blow
    # them up across the full position grid, so those keep the vmapped path.
    if (
        out_idx is not None
        and isinstance(out_idx, np.ndarray)
        and FunctionFlags.SPARSE_JHJ not in flags
    ):
        lapl_args = _materialize_args_for_idx(lapl_args, out_idx, in_axes)
        out_idx = None

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
