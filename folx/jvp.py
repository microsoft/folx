import functools
import logging
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.core import Tracer

from .ad import vjp
from .api import (
    JAC_DIM,
    Array,
    Axes,
    ExtraArgs,
    ForwardFn,
    FunctionFlags,
    FwdJacobian,
    FwdLaplArgs,
    FwdLaplArray,
    MergeFn,
)
from .tree_utils import tree_concat, tree_expand, tree_take
from .utils import (
    broadcast_dim,
    broadcast_except,
    broadcast_mask_to_jacobian,
    compact_repeated_dims_except,
    extend_jacobians,
    get_jacobian_for_reduction,
    materialize_by_gather,
    np_concatenate_brdcast,
)


def _factorized_dense_row_sum(data, x0_idx, red_axes, out_dim):
    """Sums sparse Jacobian rows of a reduction into a dense Jacobian.

    Each row is first reduced densely over the reduced axes where its target
    index is constant; only the small varying remainder is materialized via
    static gathers.

    Args:
        data: Jacobian rows, shape ``(k, *x_shape)``.
        x0_idx: Static target indices, broadcastable to ``data.shape``.
        red_axes: Reduced axes in ``x_shape`` coordinates.
        out_dim: Number of dense output rows.
    Returns:
        Array of shape ``(out_dim, *kept_shape)`` or ``None`` if a remainder
        has too high duplicate multiplicity.
    """
    k, *x_shape = data.shape
    tmap = np.broadcast_to(x0_idx, data.shape)
    x_ndim = len(x_shape)
    kept = tuple(a for a in range(x_ndim) if a not in red_axes)
    kept_size = np.prod([x_shape[a] for a in kept], dtype=int)

    groups: dict[tuple[int, ...], list[int]] = {}
    for kk in range(k):
        row = tmap[kk]
        sig = tuple(a for a in red_axes if not (np.take(row, [0], axis=a) == row).all())
        groups.setdefault(sig, []).append(kk)

    total = None
    for sig, rows in groups.items():
        const_ax = tuple(a for a in red_axes if a not in sig)
        part = data[np.array(rows)].sum(tuple(a + 1 for a in const_ax))
        t = tmap[np.array(rows)]
        for a in sorted(const_ax, reverse=True):
            t = np.take(t, 0, axis=a + 1)
        # part/t: (g, *remaining axes in original order); split into kept | sig.
        remaining = [a for a in range(x_ndim) if a not in const_ax]
        kept_pos = [1 + remaining.index(a) for a in kept]
        sig_pos = [1 + remaining.index(a) for a in sig]
        order = (*kept_pos, 0, *sig_pos)
        rows2 = len(rows) * np.prod([x_shape[a] for a in sig], dtype=int)
        part = jnp.transpose(part, order).reshape(kept_size, rows2)
        t = np.transpose(t, order).reshape(kept_size, rows2)
        gathered = materialize_by_gather(part, t, out_dim)
        if gathered is None:
            return None
        total = gathered if total is None else total + gathered
    if total is None:
        return None
    return jnp.moveaxis(total, 1, 0).reshape(out_dim, *[x_shape[a] for a in kept])


def sparse_sum_jvp(
    laplace_args: FwdLaplArgs,
    axes: Axes,
    kwargs,
    sparsity_threshold: int,
):
    x = laplace_args.x[0]
    x_lapl = laplace_args.laplacian[0]
    x_jac = laplace_args.jacobian[0]

    if axes is None:
        axes = kwargs.get('axes')
    if axes is None:
        axes = tuple(range(x.ndim))

    # these are fairly easy to compute
    y = x.sum(axes)
    y_lapl = x_lapl.sum(axes)

    # for the sparse jacobian, we will use a segment sum
    out_shape = y.shape
    out_size = np.prod(out_shape, dtype=int)
    reduced_dims = (JAC_DIM,) + tuple(i + (i >= JAC_DIM) for i in axes)
    non_reduced_axes = tuple(i for i in range(x_jac.ndim) if i not in reduced_dims)
    assert x_jac.x0_idx is not None
    axes_order = reduced_dims + non_reduced_axes

    def compute_outdeps(arr: np.ndarray, axis: int):
        A_sorted = np.sort(arr, axis=axis)
        max_out = (np.diff(A_sorted, axis=axis) > 0).sum(axis).max() + 1
        # move axis to back so we can use vectorize
        A_sorted = np.moveaxis(A_sorted, axis, -1)
        with jax.ensure_compile_time_eval():
            unique = functools.partial(jnp.unique, size=max_out, fill_value=-1)
            unique = jnp.vectorize(unique, signature='(n)->(m)')
            idx_out = unique(A_sorted)
        idx_out = np.moveaxis(np.asarray(idx_out), -1, axis)
        return idx_out

    # Create output mask
    idx = np.transpose(x_jac.x0_idx, axes_order).reshape(-1, out_size)
    idx_out = compute_outdeps(idx, axis=0)
    if idx_out.shape[0] > sparsity_threshold:
        logging.info(
            f'Output ({idx_out.shape[0]}) reaches sparsity threshold ({sparsity_threshold}). Switching to dense.'
        )
        out_dim = np.max(idx) + 1
        canonical_axes = tuple(a % x.ndim for a in axes)
        out_jac = _factorized_dense_row_sum(
            x_jac.data, x_jac.x0_idx, canonical_axes, out_dim
        )
        if out_jac is not None:
            return y, FwdJacobian(out_jac, None), y_lapl
        idx_out = None
    else:
        idx = np.argmax(idx[:, None] == idx_out, axis=1)
        out_dim = idx_out.shape[0]
        idx_out = idx_out.reshape(out_dim, *out_shape)

    # segment sum on the jacobian
    jac = jnp.transpose(x_jac.data, axes_order).reshape(-1, out_size)
    # While we could just stop here and use a vmapped segment sum elementwise, it is more efficient
    # to reduce the mask along repeated dimensions and do block segment sums.
    jac = jac.reshape(-1, *out_shape)
    idx = idx.reshape(-1, *out_shape)
    idx, repeated_dims = compact_repeated_dims_except(idx, 0)
    vmapped_axes = tuple(i for i in range(idx.ndim) if i not in repeated_dims and i > 0)
    new_order = (0, *vmapped_axes, *repeated_dims)
    inv_order = np.argsort(new_order)
    idx = np.transpose(idx, new_order)[(..., *(0,) * len(repeated_dims))]
    jac = jnp.transpose(jac, new_order)
    jac_in_shape = jac.shape
    idx = idx.reshape(idx.shape[0], -1)
    jac = jac.reshape(*idx.shape[:2], -1)
    # Compute output: static gather when possible, runtime scatter otherwise.
    gathered = None
    if isinstance(idx, np.ndarray):
        gathered = materialize_by_gather(jnp.moveaxis(jac, 0, 1), idx.T, out_dim)
    if gathered is not None:
        out_jac = jnp.moveaxis(gathered, 0, 1)
    else:
        seg_sum = functools.partial(jax.ops.segment_sum, num_segments=out_dim)
        out_jac = jax.vmap(seg_sum, in_axes=1, out_axes=1)(jac, idx)
    # reshape back
    out_jac = out_jac.reshape(out_dim, *jac_in_shape[1:])
    out_jac = np.transpose(out_jac, inv_order)
    out_jac = out_jac.reshape(out_dim, *out_shape)
    return y, FwdJacobian(out_jac, idx_out), y_lapl


def sparse_jvp(
    fwd: ForwardFn,
    laplace_args: FwdLaplArgs,
    extra_args: ExtraArgs,
    merge: MergeFn,
    axes: Axes,
    kwargs,
    sparsity_threshold: int,
    flags: FunctionFlags,
    in_axes: Axes,
) -> tuple[Array, FwdJacobian, Array]:
    if not laplace_args.all_jacobian_weak:
        return dense_jvp(fwd, laplace_args, in_axes=in_axes)

    # Scatter ops (before the axes logic since operand, indices and updates
    # generally differ in ndim)
    if FunctionFlags.SCATTER in flags:
        return sparse_scatter_jvp(
            fwd,
            laplace_args,
            extra_args,
            merge,
            flags=flags,
            in_axes=in_axes,
            kwargs=kwargs,
            sparsity_threshold=sparsity_threshold,
        )

    if axes is None:
        axes = kwargs.get('axes')
    if axes is None:
        ndims = set(x.ndim for x in laplace_args.x)
        if len(ndims) != 1:
            return dense_jvp(fwd, laplace_args, in_axes=in_axes)
        axes = tuple(range(next(iter(ndims))))
    if isinstance(axes, int):
        axes = (axes,)

    # Elementwise ops
    if axes == () or np.array(axes).size == 0:
        return sparse_diag_jvp(fwd, laplace_args, in_axes=in_axes)

    # Summation
    if FunctionFlags.SUMMATION in flags:
        return sparse_sum_jvp(laplace_args, axes, kwargs, sparsity_threshold)

    grad_tan, out_mask = get_jacobian_for_reduction(laplace_args.jacobian, axes)
    if out_mask.shape[JAC_DIM] > sparsity_threshold:
        logging.info(
            f'Output ({out_mask.shape[JAC_DIM]}) reaches sparsity threshold ({sparsity_threshold}). Switching to dense.'
        )
        return dense_jvp(fwd, laplace_args, in_axes=in_axes)

    tangent = tree_concat(
        broadcast_except(
            [grad_tan, tree_expand(laplace_args.laplacian, axis=JAC_DIM)], JAC_DIM
        ),
        axis=JAC_DIM,
    )

    @functools.partial(jax.vmap, in_axes=0, out_axes=(None, 0))
    def jvp(tangents):
        return jax.jvp(fwd, laplace_args.x, tangents)

    y, y_tangent = jvp(tangent)
    grad_y = tree_take(y_tangent, slice(None, -1), axis=JAC_DIM)
    lapl_y = tree_take(y_tangent, -1, axis=JAC_DIM)

    new_masks = broadcast_mask_to_jacobian(out_mask, grad_y)
    assert jtu.tree_all(
        jtu.tree_map(lambda g, m: g.shape == m.shape, grad_y, new_masks)
    )

    grad_y = jtu.tree_map(FwdJacobian, grad_y, new_masks)
    return y, grad_y, lapl_y


def sparse_diag_jvp(
    fwd: ForwardFn, laplace_args: FwdLaplArgs, in_axes: Axes
) -> tuple[Array, FwdJacobian, Array]:
    if not laplace_args.all_jacobian_weak:
        return dense_jvp(fwd, laplace_args, in_axes=in_axes)

    y = fwd(*laplace_args.x)
    if (
        isinstance(y, Array)
        and len(laplace_args) == 1
        and y.shape == laplace_args.x[0].shape
    ):
        # If we have elementwise functions, we can just compute the full jacobian and
        # do the operations a bit faster.
        jac = vjp(fwd, laplace_args.x[0])(jnp.ones_like(y))[0]
        grad_y = jac * laplace_args.jacobian[0].data
        lapl_y = jac * laplace_args.laplacian[0]
    else:
        # for diagonal operations we must use one hot encoded jacobians, i.e.,
        # all but one will be zero for the jvp and we repeat this for all jacobians.
        # After we check which jacobians use the same mask, these we sum.
        # The different masks will be concatenated.
        tangent = tree_concat(
            [
                *laplace_args.one_hot_sparse_jacobian,
                tree_expand(laplace_args.laplacian, axis=JAC_DIM),
            ],
            axis=JAC_DIM,
        )

        @functools.partial(jax.vmap, in_axes=0, out_axes=(None, 0))
        def jvp(tangents):
            return jax.jvp(fwd, laplace_args.x, tangents)

        y, y_tangent = jvp(tangent)
        grad_y = tree_take(y_tangent, slice(None, -1), axis=JAC_DIM)
        lapl_y = tree_take(y_tangent, -1, axis=JAC_DIM)

    if len(laplace_args) == 1:
        # If we only have a single argument, we can safe some time by not
        # doing the segment sum as the output mask will be the same as the input mask.
        result_mask = laplace_args.jacobian_mask[0]
    else:
        # Compute the resulting masks and the associated index tensors
        result_mask, inv = np.unique(
            np_concatenate_brdcast(laplace_args.jacobian_mask, axis=JAC_DIM),
            axis=JAC_DIM,
            return_inverse=True,
        )
        # Merge rows sharing a mask. The mapping is static, so a gather plus
        # per-group sums beats a runtime scatter (segment_sum).
        inv = inv.reshape(-1)
        groups = [np.where(inv == o)[0] for o in range(result_mask.shape[JAC_DIM])]

        def merge_rows(g):
            parts = [
                g[group].sum(JAC_DIM, keepdims=True) if group.size > 1 else g[group]
                for group in groups
            ]
            return jnp.concatenate(parts, axis=JAC_DIM) if len(parts) > 1 else parts[0]

        grad_y = jtu.tree_map(merge_rows, grad_y)

    # We need to broadcast the output mask to the shape of the gradient in case the operation
    # included some broadcasting, e.g., (10, 1) * (5,) -> (10, 5)
    result_mask = broadcast_mask_to_jacobian(result_mask, grad_y)
    assert jtu.tree_all(
        jtu.tree_map(lambda g, m: g.shape == m.shape, grad_y, result_mask)
    )

    grad_y = jtu.tree_map(FwdJacobian, grad_y, result_mask)
    return y, grad_y, lapl_y


def sparse_index_jvp(
    fwd_fn: ForwardFn,
    merged_fwd: ForwardFn,
    laplace_args: FwdLaplArgs,
    extra_args: ExtraArgs,
    merge: MergeFn,
    index_static_args: tuple | slice | None,
    in_axes: Axes,
) -> tuple[Array, FwdJacobian, Array]:
    # For indexing operations we have to also index the mask, here we can just apply the jacobian
    if not laplace_args.all_jacobian_weak:
        return dense_jvp(merged_fwd, laplace_args, in_axes=in_axes)

    # Compute output mask
    try:
        # We must disable any jit tracer and evaluate the index operation
        # explictly here. This allows us to perform the index operation on our mask.
        # An index operation is expected to be static. If it is not, we will default to
        # materializing everything.
        # https://github.com/google/jax/pull/3370
        with jax.ensure_compile_time_eval():
            extra_filled = jtu.tree_map(
                lambda x: jnp.full(
                    x.shape if isinstance(x, jax.Array) else (), -1, dtype=jnp.int32
                ),
                extra_args,
            )

            def _merged_fwd(*args):
                # For index operations some operands may be static, i.e., they are not
                # part of the output. We need to make sure that we do not fill these.
                non_filled_args = merge(args, extra_args)
                if index_static_args is None:
                    return fwd_fn(*non_filled_args)
                static_idx = index_static_args
                filled_args = merge(args, extra_filled)
                if isinstance(static_idx, slice):
                    all_idx = np.arange(len(filled_args))
                    static_idx = all_idx[static_idx]
                return fwd_fn(
                    *[
                        (non_filled_args[i] if i in static_idx else filled_args[i])
                        for i in range(len(filled_args))
                    ]
                )  # type: ignore

            # Masks must be int32 to match the fill values of constant
            # arguments; under x64 they may arrive as int64.
            masks = jtu.tree_map(
                lambda m: np.asarray(m, dtype=np.int32),
                laplace_args.jacobian_mask,
            )
            mask = jax.vmap(_merged_fwd, in_axes=JAC_DIM, out_axes=JAC_DIM)(
                *broadcast_dim(masks, fill_value=-1, axis=JAC_DIM)
            )
            mask = jtu.tree_map(lambda x: np.asarray(x, dtype=int), mask)
    except Exception as e:
        logging.warning(
            f'Could not perform index operation {fwd_fn.__name__}. '
            'This is most likely due to data dependent indexing. '
            'We will default to materializing everything. Here is the caught exception:\n'
            f'{e}'
        )
        return dense_jvp(merged_fwd, laplace_args, in_axes=in_axes)

    tangent = tree_concat(
        [
            broadcast_dim(laplace_args.sparse_jacobian, fill_value=0, axis=JAC_DIM),
            tree_expand(laplace_args.laplacian, axis=JAC_DIM),
        ],
        axis=JAC_DIM,
    )

    @functools.partial(jax.vmap, in_axes=0, out_axes=(None, 0))
    def jvp(tangents):
        return jax.jvp(merged_fwd, laplace_args.x, tangents)

    y, y_tangent = jvp(tangent)
    grad_y = tree_take(y_tangent, slice(None, -1), axis=JAC_DIM)
    lapl_y = tree_take(y_tangent, -1, axis=JAC_DIM)

    assert jtu.tree_all(jtu.tree_map(lambda a, b: a.shape == b.shape, grad_y, mask))
    grad_y = jtu.tree_map(FwdJacobian, grad_y, mask)
    return y, grad_y, lapl_y


def _scatter_target_positions(
    op_shape: tuple[int, ...],
    scatter_indices: Array,
    updates_shape: tuple[int, ...],
    kwargs,
) -> np.ndarray:
    """Computes the flat operand position written by each updates element.

    Gathers per-dimension operand coordinates through the VJP of a scatter-add
    with the same indices and dimension numbers, which is exactly the
    transposed gather. This delegates all scatter semantics (window mapping,
    batching dims, out-of-bounds handling) to JAX and is evaluated at compile
    time.

    Args:
        op_shape: Shape of the scatter operand.
        scatter_indices: Static scatter index array.
        updates_shape: Shape of the scatter updates.
        kwargs: Parameters of the scatter primitive.
    Returns:
        Integer array of shape ``updates_shape`` with flat operand positions;
        ``-1`` marks dropped (out-of-bounds) updates.
    """
    with jax.ensure_compile_time_eval():

        def scatter_zeros(u):
            return jax.lax.scatter_add(
                jnp.zeros(op_shape, jnp.float32),
                scatter_indices,
                u,
                kwargs['dimension_numbers'],
                indices_are_sorted=kwargs.get('indices_are_sorted', False),
                unique_indices=kwargs.get('unique_indices', False),
                mode=kwargs.get('mode'),
            )

        _, vjp_fn = jax.vjp(scatter_zeros, jnp.zeros(updates_shape, jnp.float32))
        # First channel detects dropped updates, the rest carry coordinates.
        channels = [jnp.ones(op_shape, jnp.float32)]
        for d, size in enumerate(op_shape):
            shape = [1] * len(op_shape)
            shape[d] = size
            channels.append(
                jnp.broadcast_to(
                    jnp.arange(size, dtype=jnp.float32).reshape(shape), op_shape
                )
            )
        gathered = np.asarray(jax.vmap(lambda c: vjp_fn(c)[0])(jnp.stack(channels)))
    valid = gathered[0] > 0.5
    positions = np.full(updates_shape, -1, dtype=np.int64)
    if len(op_shape) == 0:
        positions[valid] = 0
        return positions
    coords = tuple(np.round(c[valid]).astype(np.int64) for c in gathered[1:])
    positions[valid] = np.ravel_multi_index(coords, op_shape)
    return positions


def sparse_scatter_jvp(
    fwd: ForwardFn,
    laplace_args: FwdLaplArgs,
    extra_args: ExtraArgs,
    merge: MergeFn,
    flags: FunctionFlags,
    in_axes: Axes,
    kwargs,
    sparsity_threshold: int,
) -> tuple[Array, FwdJacobian, Array]:
    """JVP for scatter ops (set/add/min/max) with sparse Jacobians.

    Statically computes the target operand position of every updates element,
    unions the dependencies of operand and updates per output position, and
    runs the scatter's JVP once per output mask row with all Jacobians aligned
    to that mask. For scatter_add (linear with unit coefficients) the output
    Jacobian is instead accumulated directly from the sparse input rows,
    which avoids materializing the aligned tangents entirely.

    Args:
        fwd: Scatter forward function taking only the ``FwdLaplArray`` args.
        laplace_args: The ``FwdLaplArray`` arguments (operand and/or updates).
        extra_args: Constant arguments.
        merge: Function merging ``laplace_args`` and ``extra_args`` into
            ``(operand, scatter_indices, updates)``.
        flags: Function flags of the scatter op.
        in_axes: Input axes of the op.
        kwargs: Parameters of the scatter primitive.
        sparsity_threshold: Maximum sparse output rows before densifying.
    Returns:
        Tuple of output, output Jacobian and output Laplacian.
    """
    operand, scatter_indices, updates = merge(laplace_args.arrays, extra_args)  # type: ignore
    if isinstance(scatter_indices, (FwdLaplArray, Tracer)):
        return dense_jvp(fwd, laplace_args, in_axes=in_axes)
    # scatter (set) overwrites operand entries; add/min/max keep them varying.
    is_set = FunctionFlags.INDEXING in flags
    is_add = fwd.__name__ == 'scatter_add'
    if not isinstance(updates, FwdLaplArray) and is_add:
        # Adding constants only shifts the output; the operand must be a
        # FwdLaplArray by exclusion and its Jacobian and Laplacian pass through.
        y: Array = fwd(*laplace_args.x)  # type: ignore
        return y, operand.jacobian, operand.laplacian  # type: ignore

    op_shape = tuple(operand.shape)
    op_size = int(np.prod(op_shape, dtype=int))
    try:
        target_pos = _scatter_target_positions(
            op_shape, scatter_indices, tuple(updates.shape), kwargs
        )
    except Exception as e:
        logging.warning(
            f'Could not compute static scatter positions for {fwd.__name__}. '
            'This is most likely due to data dependent indexing. '
            'We will default to materializing everything. Here is the caught exception:\n'
            f'{e}'
        )
        return dense_jvp(fwd, laplace_args, in_axes=in_axes)

    # Union the input dependencies per output position.
    pos_list, dep_list = [], []
    if isinstance(updates, FwdLaplArray):
        mask = updates.jacobian.mask
        mask = np.broadcast_to(mask, (mask.shape[JAC_DIM], *updates.shape))
        tpos = np.broadcast_to(target_pos, mask.shape)
        select = (mask >= 0) & (tpos >= 0)
        pos_list.append(tpos[select])
        dep_list.append(mask[select])
    if isinstance(operand, FwdLaplArray):
        mask = operand.jacobian.mask
        mask = np.broadcast_to(mask, (mask.shape[JAC_DIM], *op_shape))
        keep = mask >= 0
        if is_set:
            # Overwritten positions no longer depend on the operand.
            overwritten = np.zeros((op_size,), dtype=bool)
            overwritten[target_pos[target_pos >= 0]] = True
            keep &= ~overwritten.reshape(op_shape)
        pos = np.broadcast_to(np.arange(op_size).reshape(op_shape), mask.shape)
        pos_list.append(pos[keep])
        dep_list.append(mask[keep])

    pos = np.concatenate(pos_list).astype(np.int64)
    dep = np.concatenate(dep_list).astype(np.int64)
    n_dep = int(dep.max()) + 1 if dep.size > 0 else 1
    unique_keys = np.unique(pos * n_dep + dep)
    pos_u, dep_u = unique_keys // n_dep, unique_keys % n_dep
    counts = np.bincount(pos_u, minlength=op_size)
    max_out = int(counts.max()) if counts.size > 0 else 0
    # scatter_add is linear with unit coefficients, so the output Jacobian can
    # be accumulated straight from the sparse input rows (k_upd many) instead
    # of aligning max_out tangent rows per argument. Only worth it when the
    # saved work is substantial; for small scatters the aligned path fuses
    # into fewer kernels.
    k_upd = (
        updates.jacobian.mask.shape[JAC_DIM]
        if isinstance(updates, FwdLaplArray)
        else max_out
    )
    use_direct = (
        is_add
        and isinstance(updates, FwdLaplArray)
        and (max_out - k_upd) * int(np.prod(updates.shape, dtype=int)) > 2**16
    )
    densify = max_out > sparsity_threshold
    if densify:
        # Densifying the small scatter output is often much cheaper than
        # materializing the inputs' dense Jacobians before the scatter.
        n_dense = max(a.jacobian.max_n for a in laplace_args.arrays) + 1
        upd_size = int(np.prod(updates.shape, dtype=int))
        if use_direct:
            sparse_cost = k_upd * upd_size + (max_out + n_dense) * op_size
        else:
            sparse_cost = (max_out + 1) * (upd_size + op_size) + n_dense * op_size
        dense_cost = n_dense * (upd_size + op_size)
        if sparse_cost >= dense_cost:
            logging.info(
                f'Scatter: Output ({max_out}) reaches sparsity threshold ({sparsity_threshold}). Switching to dense.'
            )
            return dense_jvp(fwd, laplace_args, in_axes=in_axes)
        logging.info(
            f'Scatter: Output ({max_out}) reaches sparsity threshold ({sparsity_threshold}). Densifying after the scatter.'
        )
    max_out = max(max_out, 1)
    out_mask_flat = np.full((max_out, op_size), -1, dtype=np.int32)
    group_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slots = np.arange(pos_u.size) - group_starts[pos_u]
    out_mask_flat[slots, pos_u] = dep_u
    out_mask = out_mask_flat.reshape(max_out, *op_shape)

    # Identify which laplace args are the operand and the updates.
    arg_ids = merge(tuple(range(len(laplace_args))), (None,) * len(extra_args))  # type: ignore

    def to_jacobian(g):
        jac = FwdJacobian(g, out_mask)
        return jac.as_dense if densify else jac

    if use_direct:
        grad_out = None
        for i, arr in enumerate(laplace_args.arrays):
            if i == arg_ids[2]:
                # Updates: bucket every sparse entry by (output row, position).
                mask = np.broadcast_to(
                    arr.jacobian.mask, (k_upd, *updates.shape)
                ).astype(np.int64)
                tpos = np.broadcast_to(target_pos, mask.shape)
                key = tpos * n_dep + mask
                idx = np.minimum(
                    np.searchsorted(unique_keys, key), unique_keys.size - 1
                )
                found = (mask >= 0) & (tpos >= 0) & (unique_keys[idx] == key)
                bucket = np.where(found, slots[idx] * op_size + tpos, -1).reshape(-1)
                data = jnp.broadcast_to(arr.jacobian.data, mask.shape).reshape(-1)
                gathered = materialize_by_gather(
                    data[None], bucket[None], max_out * op_size
                )
                if gathered is not None:
                    contrib = gathered[0]
                else:
                    contrib = jax.ops.segment_sum(data, bucket, max_out * op_size)
                contrib = contrib.reshape(max_out, *op_shape)
            else:
                # The operand passes through scatter_add unchanged.
                contrib = arr.jacobian.materialize_for_idx(
                    arr.jacobian.get_index_mask(out_mask), max_idx=max_out
                )
                contrib = jnp.broadcast_to(contrib, (max_out, *op_shape))
            grad_out = contrib if grad_out is None else grad_out + contrib
        lapl_tangents = tuple(
            jnp.broadcast_to(a.laplacian, a.shape) for a in laplace_args.arrays
        )
        y, lapl_y = jax.jvp(fwd, laplace_args.x, lapl_tangents)
        return y, to_jacobian(grad_out), lapl_y

    # Align every argument's Jacobian rows with the output mask rows.
    tangents = []
    for i, arr in enumerate(laplace_args.arrays):
        if i == arg_ids[2]:  # updates
            outputs = out_mask_flat[:, np.maximum(target_pos, 0)]
            outputs = np.where(target_pos >= 0, outputs, -2)
        else:  # operand
            outputs = out_mask
        grad_tan = arr.jacobian.materialize_for_idx(
            arr.jacobian.get_index_mask(outputs), max_idx=max_out
        )
        grad_tan = jnp.broadcast_to(grad_tan, (max_out, *arr.shape))
        lapl = jnp.broadcast_to(arr.laplacian, arr.shape)
        tangents.append(
            jnp.concatenate([grad_tan, jnp.expand_dims(lapl, JAC_DIM)], axis=JAC_DIM)
        )

    @functools.partial(jax.vmap, in_axes=0, out_axes=(None, 0))
    def jvp(tangents):
        return jax.jvp(fwd, laplace_args.x, tangents)

    y, y_tangent = jvp(tuple(tangents))
    grad_y = tree_take(y_tangent, slice(None, -1), axis=JAC_DIM)
    lapl_y = tree_take(y_tangent, -1, axis=JAC_DIM)

    grad_y = jtu.tree_map(to_jacobian, grad_y)
    return y, grad_y, lapl_y


def dense_split_jvp(
    fwd: ForwardFn, laplace_args: FwdLaplArgs
) -> tuple[Array, FwdJacobian, Array]:
    y, jvp = jax.linearize(fwd, *laplace_args.x)
    grad_y = jax.vmap(jvp)(
        *extend_jacobians(*laplace_args.dense_jacobian, axis=JAC_DIM)
    )
    lapl_y = jvp(*laplace_args.laplacian)
    return y, grad_y, lapl_y


def dense_elementwise_jvp(
    fwd: ForwardFn, laplace_args: FwdLaplArgs
) -> tuple[Array, FwdJacobian, Array]:
    y: Array = fwd(laplace_args.x[0])  # type: ignore
    if not isinstance(y, Array) or y.shape != laplace_args.x[0].shape:
        return dense_split_jvp(fwd, laplace_args)

    jac = vjp(fwd, laplace_args.x[0])(jnp.ones_like(y))[0]
    grad_y = jac * laplace_args.dense_jacobian[0]
    lapl_y = jac * laplace_args.laplacian[0]
    return y, grad_y, lapl_y


def dense_jvp(
    fwd: ForwardFn,
    laplace_args: FwdLaplArgs,
    in_axes: Axes,
) -> tuple[Array, FwdJacobian, Array]:
    # General implementation. This will always materialize the full Jacobian.
    if in_axes == () and len(laplace_args) == 1:
        y, grad_y, lapl_y = dense_elementwise_jvp(fwd, laplace_args)
    else:
        y, grad_y, lapl_y = dense_split_jvp(fwd, laplace_args)
    grad_y = jtu.tree_map(FwdJacobian.from_dense, grad_y)
    return y, grad_y, lapl_y


def get_jvp_function(
    fwd: ForwardFn,
    flags: FunctionFlags,
    in_axes: Axes,
    extra_args: ExtraArgs,
    merge: MergeFn,
    index_static_args: tuple | slice | None,
    sparsity_threshold: int,
):
    def merged_fwd(*args: Array):
        return fwd(*merge(args, extra_args))

    merged_fwd.__name__ = fwd.__name__

    def parallel_jvp(args: FwdLaplArgs, kwargs):
        if not args.all_jacobian_weak:
            return dense_jvp(merged_fwd, args, in_axes)
        if FunctionFlags.INDEXING in flags:
            return sparse_index_jvp(
                fwd,
                merged_fwd,
                args,
                extra_args,
                merge,
                index_static_args,
                in_axes=in_axes,
            )
        return sparse_jvp(
            merged_fwd,
            args,
            extra_args,
            merge,
            axes=in_axes,
            kwargs=kwargs,
            sparsity_threshold=sparsity_threshold,
            flags=flags,
            in_axes=in_axes,
        )

    def one_by_one_jvp(args: FwdLaplArgs, kwargs) -> tuple[Array, FwdJacobian, Array]:
        y, grad, lapl = None, None, None
        for i, x in enumerate(args.arrays):
            static_args = tuple(args.x)
            n_static = len(static_args) - 1
            new_extra = extra_args + static_args[:i] + static_args[i + 1 :]

            def new_merge(args: Sequence[Array], extra: Sequence[Array]):
                assert len(args) == 1, 'Only one argument is expected.'
                extra, static = extra[:-n_static], extra[-n_static:]
                return merge(tuple(static[:i] + (args[0],) + static[i:]), extra)

            def merged_fwd(*args: Array):
                return fwd(*new_merge(args, new_extra))

            merged_fwd.__name__ = fwd.__name__

            def _jvp(args: FwdLaplArgs, kwargs):
                # logging.info(f'{vmapped_jvp.__name__} {args.arrays[0].jacobian.data.shape}')
                # If any jacobian is dense, we just switch all jacobians to dense.
                if not args.all_jacobian_weak:
                    return dense_jvp(merged_fwd, args, in_axes)

                # Special case for index operation
                return sparse_jvp(
                    merged_fwd,
                    args,
                    new_extra,
                    new_merge,
                    axes=in_axes,
                    kwargs=kwargs,
                    sparsity_threshold=sparsity_threshold,
                    flags=flags,
                    in_axes=in_axes,
                )

            y_, grad_, lapl_ = _jvp(FwdLaplArgs((x,)), kwargs)
            if y is None:
                y, grad, lapl = y_, grad_, lapl_
            else:
                # For multi-output functions grad and lapl are pytrees, so we
                # accumulate per output rather than via += (tuple concatenation).
                grad = jtu.tree_map(
                    lambda a, b: a + b,
                    grad,
                    grad_,
                    is_leaf=lambda a: isinstance(a, FwdJacobian),
                )
                lapl = jtu.tree_map(lambda a, b: a + b, lapl, lapl_)
        return y, grad, lapl  # type: ignore

    def jvp(args: FwdLaplArgs, kwargs) -> tuple[Array, FwdJacobian, Array]:
        # If everything is dense, we do it in parallel. Otherwise, we call the simpler code
        # if only a single argument has a jacobian or it is an elementwise/indexing/scatter
        # operation.
        if (not args.any_jacobian_weak) or (
            args.all_jacobian_weak
            and (
                (FunctionFlags.INDEXING in flags)
                or (FunctionFlags.SCATTER in flags)
                or (in_axes == ())
                or (len(args) == 1)
            )
        ):
            return parallel_jvp(args, kwargs)
        else:
            return one_by_one_jvp(args, kwargs)

    return jvp
