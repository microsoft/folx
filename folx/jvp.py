import functools
import logging
from typing import TypeVar

import jax
import jax.core as core
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

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
    PyTree,
)
from .tree_utils import tree_concat, tree_expand, tree_take
from .utils import (
    broadcast_except,
    broadcast_dim,
    extend_jacobians,
    get_jacobian_for_reduction,
    np_concatenate_brdcast,
)

R = TypeVar('R', bound=PyTree[Array])


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
        return dense_jvp(fwd, laplace_args, flags=flags, in_axes=in_axes)

    if axes is None:
        if 'axes' in kwargs:
            axes = kwargs['axes']
        elif 'axis' in kwargs:
            axes = kwargs['axis']
    if axes is None:
        axes = tuple(range(laplace_args.arrays[0].x.ndim))
    if isinstance(axes, int):
        axes = (axes,)
    if axes == () or np.array(axes).size == 0:
        return sparse_diag_jvp(fwd, laplace_args, flags=flags, in_axes=in_axes)

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

    # TODO: One could also do the reduction after the jvp. In some cases that's more efficient.
    grad_tan, out_mask = get_jacobian_for_reduction(laplace_args.jacobian, axes)
    if out_mask.shape[JAC_DIM] > sparsity_threshold:
        logging.info(
            f'Output ({out_mask.shape[JAC_DIM]}) reaches sparsity threshold ({sparsity_threshold}). Switching to dense.'
        )
        return dense_jvp(fwd, laplace_args, flags=flags, in_axes=in_axes)

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

    new_masks = jtu.tree_map(lambda m, g: np.broadcast_to(m, g.shape), out_mask, grad_y)
    assert grad_y.shape == new_masks.shape, f'{grad_y.shape} != {new_masks.shape}'

    grad_y = jtu.tree_map(FwdJacobian, grad_y, new_masks)
    return y, grad_y, lapl_y


def sparse_diag_jvp(
    fwd: ForwardFn, laplace_args: FwdLaplArgs, flags: FunctionFlags, in_axes: Axes
) -> tuple[Array, FwdJacobian, Array]:
    if not laplace_args.all_jacobian_weak:
        return dense_jvp(fwd, laplace_args, flags=flags, in_axes=in_axes)

    y = fwd(*laplace_args.x)
    if (
        isinstance(y, Array)
        and len(laplace_args) == 1
        and y.shape == laplace_args.x[0].shape
    ):
        # If we have elementwise functions, we can just compute the full jacobian and
        # do the operations a bit faster.
        jac = jax.grad(lambda x: jnp.sum(fwd(x)))(laplace_args.x[0])  # type: ignore
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
        # Aggergate the corresponding jacobian results. Unfortunately,
        # segment_sum can only be applied over the leading axis. So, we have to
        # do some rearranging here.
        grad_y = jax.ops.segment_sum(
            grad_y, inv, num_segments=result_mask.shape[JAC_DIM]
        )

    # We need to broadcast the output mask to the shape of the gradient in case the operation
    # included some broadcasting, e.g., (10, 1) * (5,) -> (10, 5)
    result_mask = jtu.tree_map(
        lambda m, g: np.broadcast_to(m, g.shape), result_mask, grad_y
    )
    assert grad_y.shape == result_mask.shape, f'{grad_y.shape} != {result_mask.shape}'

    grad_y = jtu.tree_map(FwdJacobian, grad_y, result_mask)
    return y, grad_y, lapl_y


def sparse_index_jvp(
    fwd_fn: ForwardFn,
    merged_fwd: ForwardFn,
    laplace_args: FwdLaplArgs,
    extra_args: ExtraArgs,
    merge: MergeFn,
    index_static_args: tuple | slice | None,
    flags: FunctionFlags,
    in_axes: Axes,
) -> tuple[Array, FwdJacobian, Array]:
    # For indexing operations we have to also index the mask, here we can just apply the jacobian
    if not laplace_args.all_jacobian_weak:
        return dense_jvp(merged_fwd, laplace_args, flags=flags, in_axes=in_axes)

    # Compute output mask
    try:
        # We must disable any jit tracer and evaluate the index operation
        # explictly here. This allows us to perform the index operation on our mask.
        # An index operation is expected to be static. If it is not, we will default to
        # materializing everything.
        # https://github.com/google/jax/pull/3370
        with core.new_main(core.EvalTrace, dynamic=True):
            extra_filled = jtu.tree_map(
                lambda x: jnp.full(x.shape, -1, dtype=jnp.int32), extra_args
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

            mask = jax.vmap(_merged_fwd, in_axes=JAC_DIM, out_axes=JAC_DIM)(
                *broadcast_dim(laplace_args.jacobian_mask, fill_value=-1, axis=JAC_DIM)
            )
            mask = jtu.tree_map(lambda x: np.asarray(x, dtype=int), mask)
    except Exception as e:
        logging.warning(
            f'Could not perform index operation {fwd_fn.__name__}. '
            'This is most likely due to data dependent indexing. '
            'We will default to materializing everything. Here is the caught exception:\n'
            f'{e}'
        )
        return dense_jvp(merged_fwd, laplace_args, flags=flags, in_axes=in_axes)

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

    assert grad_y.shape == mask.shape
    grad_y = jtu.tree_map(FwdJacobian, grad_y, mask)
    return y, grad_y, lapl_y


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
    operand, scatter_indices, updates = merge(laplace_args.arrays, extra_args)  # type: ignore
    if isinstance(scatter_indices, FwdLaplArray):
        return dense_jvp(fwd, laplace_args, flags=flags, in_axes=in_axes)
    if not isinstance(updates, FwdLaplArray) and FunctionFlags.LINEAR in flags:
        # operand must be a fwdlapl array by exclusion since at least one has to be.
        y: Array = fwd(*laplace_args.x)  # type: ignore
        grad_y: FwdJacobian = operand.jacobian  # type: ignore
        lapl_y: Array = operand.laplacian  # type: ignore
        return y, grad_y, lapl_y
    if isinstance(operand, FwdLaplArray):
        logging.info(
            'Scatter: operation on operand not supported. At the moment only segment sums are supported.'
        )
        return dense_jvp(fwd, laplace_args, flags=flags, in_axes=in_axes)

    dimension_numbers: jax.lax.ScatterDimensionNumbers = kwargs['dimension_numbers']
    if (
        dimension_numbers.inserted_window_dims != (0,)
        or dimension_numbers.scatter_dims_to_operand_dims != (0,)
        or dimension_numbers.update_window_dims != ()
    ):
        logging.info(
            'Scatter: dimension numbers not supported. At the moment only segment sums are supported.'
        )
        return dense_jvp(fwd, laplace_args, flags=flags, in_axes=in_axes)

    updates: FwdLaplArray = updates
    n = updates.jacobian.max_n + 1
    with core.new_main(core.EvalTrace, dynamic=True):
        one_hot_mask = jax.nn.one_hot(
            updates.jacobian.x0_idx, n, axis=-1, dtype=jnp.int32
        ).sum(0)
        out_mask = np.array(jax.ops.segment_sum(one_hot_mask, scatter_indices[:, 0]))
        max_out = np.sum(out_mask.astype(bool), axis=-1).max()
        out_mask = np.where(
            out_mask > 0, out_mask * jnp.arange(n), np.iinfo(np.int32).max
        )
        unique_fn = jax.vmap(functools.partial(jnp.unique, size=max_out, fill_value=-1))
        out_mask = unique_fn(out_mask.reshape(-1, n)).T.reshape(
            max_out, *out_mask.shape[:-1]
        )
        out_mask = np.where(out_mask == np.iinfo(np.int32).max, -1, out_mask)
    if out_mask.shape[JAC_DIM] > sparsity_threshold:
        logging.info(
            f'Scatter: Output ({out_mask.shape[JAC_DIM]}) reaches sparsity threshold ({sparsity_threshold}). Switching to dense.'
        )
        return dense_jvp(fwd, laplace_args, flags=flags, in_axes=in_axes)

    grad_tan = updates.jacobian.materialize_for_idx(
        updates.jacobian.get_index_mask(
            jnp.take(out_mask, scatter_indices[:, 0], axis=1)
        ),
        max_idx=max_out,
    )

    tangent = tree_concat(
        broadcast_except(
            [(grad_tan,), tree_expand(laplace_args.laplacian, axis=JAC_DIM)], JAC_DIM
        ),
        axis=JAC_DIM,
    )

    @functools.partial(jax.vmap, in_axes=0, out_axes=(None, 0))
    def jvp(tangents):
        return jax.jvp(fwd, laplace_args.x, tangents)

    y, y_tangent = jvp(tangent)
    grad_y = tree_take(y_tangent, slice(None, -1), axis=JAC_DIM)
    lapl_y = tree_take(y_tangent, -1, axis=JAC_DIM)

    grad_y = jtu.tree_map(FwdJacobian, grad_y, out_mask)
    return y, grad_y, lapl_y


def dense_joint_jvp(
    fwd: ForwardFn,
    laplace_args: FwdLaplArgs,
) -> tuple[Array, FwdJacobian, Array]:
    # For some operation it is better to first concatenate and then do the jvp.
    tangent = tree_concat(
        [
            extend_jacobians(*laplace_args.dense_jacobian, axis=JAC_DIM),
            tree_expand(laplace_args.laplacian, axis=JAC_DIM),
        ],
        axis=JAC_DIM,
    )

    @functools.partial(jax.vmap, in_axes=0, out_axes=(None, 0))
    def jvp(tangents):
        return jax.jvp(fwd, laplace_args.x, tangents)

    y, y_tangent = jvp(tangent)
    grad_y, lapl_y = (
        tree_take(y_tangent, slice(None, -1), axis=JAC_DIM),
        tree_take(y_tangent, -1, axis=JAC_DIM),
    )
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
    if y.shape != laplace_args.x[0].shape:
        return dense_split_jvp(fwd, laplace_args)

    jac = jax.grad(lambda x: jnp.sum(fwd(x)))(laplace_args.x[0])  # type: ignore
    grad_y = jac * laplace_args.dense_jacobian[0]
    lapl_y = jac * laplace_args.laplacian[0]
    return y, grad_y, lapl_y


def dense_jvp(
    fwd: ForwardFn,
    laplace_args: FwdLaplArgs,
    flags: FunctionFlags,
    in_axes: Axes,
) -> tuple[Array, FwdJacobian, Array]:
    # General implementation. This will always materialize the full Jacobian.
    if in_axes == () and len(laplace_args) == 1:
        y, grad_y, lapl_y = dense_elementwise_jvp(fwd, laplace_args)
    elif FunctionFlags.JOIN_JVP in flags:
        y, grad_y, lapl_y = dense_joint_jvp(fwd, laplace_args)
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
            return dense_jvp(merged_fwd, args, flags, in_axes)
        if FunctionFlags.INDEXING in flags:
            return sparse_index_jvp(
                fwd,
                merged_fwd,
                args,
                extra_args,
                merge,
                index_static_args,
                flags=flags,
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
            static_args = list(args.x)

            def merged_fwd(arg: Array):
                return fwd(
                    *merge(
                        tuple(static_args[:i] + [arg] + static_args[i + 1 :]),
                        extra_args,
                    )
                )

            merged_fwd.__name__ = fwd.__name__

            def _jvp(args: FwdLaplArgs, kwargs):
                # logging.info(f'{vmapped_jvp.__name__} {args.arrays[0].jacobian.data.shape}')
                # If any jacobian is dense, we just switch all jacobians to dense.
                if not args.all_jacobian_weak:
                    return dense_jvp(merged_fwd, args, flags, in_axes)

                # Special case for index operation
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

            y_, grad_, lapl_ = _jvp(FwdLaplArgs((x,)), kwargs)
            if y is None:
                y, grad, lapl = y_, grad_, lapl_
            else:
                grad += grad_  # type: ignore
                lapl += lapl_
        return y, grad, lapl  # type: ignore

    def jvp(args: FwdLaplArgs, kwargs) -> tuple[Array, FwdJacobian, Array]:
        if (
            (not args.any_jacobian_weak)
            or (FunctionFlags.INDEXING in flags)
            or (in_axes == ())
        ):
            return parallel_jvp(args, kwargs)
        else:
            return one_by_one_jvp(args, kwargs)

    return jvp
