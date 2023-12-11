import logging
from typing import TypeVar

import jax
import jax.core as core
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .api import (JAC_DIM, Arrays, Axes, ExtraArgs, ForwardFn, FunctionFlags,
                  FwdJacobian, FwdLaplArgs, JvpFn, MergeFn, PyTree)
from .tree_utils import tree_concat, tree_idx
from .types import Array, PyTree
from .utils import (broadcast_dim, broadcast_except, extend_jacobians,
                    get_jacobian_for_reduction, np_concatenate_brdcast)

R = TypeVar("R", bound=PyTree[Array])


def sparse_reduction_jvp(
    vmapped_jvp: JvpFn, laplace_args: FwdLaplArgs, axes: Axes, kwargs, sparsity_threshold: int
):
    if not laplace_args.all_jacobian_weak:
        return general_jvp(vmapped_jvp, laplace_args)

    if axes is None:
        if "axes" in kwargs:
            axes = kwargs["axes"]
        elif "axis" in kwargs:
            axes = kwargs["axis"]
    if axes is None:
        axes = tuple(range(laplace_args.arrays[0].x.ndim))
    if isinstance(axes, int):
        axes = (axes,)
    if axes == () or np.array(axes).size == 0:
        return sparse_diag_jvp(vmapped_jvp, laplace_args)

    # TODO: One could also do the reduction after the jvp. In some cases that's more efficient.
    grad_tan, out_mask = get_jacobian_for_reduction(laplace_args.jacobian, axes)
    if out_mask.shape[JAC_DIM] > sparsity_threshold:
        logging.info(
            f"Output ({out_mask.shape[JAC_DIM]}) reaches sparsity threshold ({sparsity_threshold}). Switching to dense."
        )
        return general_jvp(vmapped_jvp, laplace_args)

    tangent = tree_concat(
        broadcast_except([grad_tan, tree_idx(laplace_args.laplacian, None)], JAC_DIM), axis=JAC_DIM
    )
    y, y_tangent = vmapped_jvp(tuple(jnp.broadcast_arrays(*laplace_args.x)), tangent)

    grad_y = tree_idx(y_tangent, slice(None, -1))
    lapl_y = tree_idx(y_tangent, -1)

    new_masks = jtu.tree_map(lambda m, g: np.broadcast_to(m, g.shape), out_mask, grad_y)
    assert grad_y.shape == new_masks.shape, f"{grad_y.shape} != {new_masks.shape}"

    grad_y = jtu.tree_map(FwdJacobian, grad_y, new_masks)
    return y, grad_y, lapl_y


def sparse_diag_jvp(
    vmapped_jvp: JvpFn,
    laplace_args: FwdLaplArgs,
):
    if not laplace_args.all_jacobian_weak:
        return general_jvp(vmapped_jvp, laplace_args)
    # for diagonal operations we must use one hot encoded jacobians, i.e.,
    # all but one will be zero for the jvp and we repeat this for all jacobians.
    # After we check which jacobians use the same mask, these we sum.
    # The different masks will be concatenated.
    tangent = tree_concat(
        [*laplace_args.one_hot_sparse_jacobian, tree_idx(laplace_args.laplacian, None)],
        axis=JAC_DIM,
    )

    # One could technically reduce the number of operations by doing this in a loop
    # and avoiding the one hot encoding. But, this would have to be done sequentially.
    y, y_tangent = vmapped_jvp(laplace_args.x, tangent)
    grad_y = tree_idx(y_tangent, slice(None, -1))
    lapl_y = tree_idx(y_tangent, -1)

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
        grad_y = jax.ops.segment_sum(grad_y, inv, num_segments=result_mask.shape[JAC_DIM])

    # We need to broadcast the output mask to the shape of the gradient in case the operation
    # included some broadcasting, e.g., (10, 1) * (5,) -> (10, 5)
    result_mask = jtu.tree_map(lambda m, g: np.broadcast_to(m, g.shape), result_mask, grad_y)
    assert grad_y.shape == result_mask.shape, f"{grad_y.shape} != {result_mask.shape}"

    grad_y = jtu.tree_map(FwdJacobian, grad_y, result_mask)
    return y, grad_y, lapl_y


def sparse_index_jvp(
    fwd_fn: ForwardFn,
    vmapped_jvp: JvpFn,
    laplace_args: FwdLaplArgs,
    extra_args: ExtraArgs,
    merge: MergeFn,
    index_static_args: tuple | slice | None,
):
    # For indexing operations we have to also index the mask, here we can just apply the jacobian
    if not laplace_args.all_jacobian_weak:
        return general_jvp(vmapped_jvp, laplace_args)

    # Compute output mask
    try:
        # We must disable any jit tracer and evaluate the index operation
        # explicitly here. This allows us to perform the index operation on our mask.
        # An index operation is expected to be static. If it is not, we will default to
        # materializing everything.
        # https://github.com/google/jax/pull/3370
        with core.new_main(core.EvalTrace, dynamic=True):
            extra_filled = jtu.tree_map(
                lambda x: jnp.full(x.shape, -1, dtype=jnp.int32), extra_args
            )

            def merged_fwd(*args):
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

            mask = jax.vmap(merged_fwd, in_axes=JAC_DIM, out_axes=JAC_DIM)(
                *broadcast_dim(laplace_args.jacobian_mask, fill_value=-1, axis=JAC_DIM)
            )
            mask = jtu.tree_map(lambda x: np.asarray(x, dtype=int), mask)
    except Exception as e:
        logging.warning(
            f"Could not perform index operation {fwd_fn.__name__}. "
            "This is most likely due to data dependent indexing. "
            "We will default to materializing everything.\n"
            f"{e}"
        )
        return general_jvp(vmapped_jvp, laplace_args)

    tangent = tree_concat(
        [
            broadcast_dim(laplace_args.sparse_jacobian, fill_value=0, axis=JAC_DIM),
            tree_idx(laplace_args.laplacian, None),
        ],
        axis=JAC_DIM,
    )

    y, y_tangent = vmapped_jvp(laplace_args.x, tangent)
    grad_y = tree_idx(y_tangent, slice(None, -1))
    lapl_y = tree_idx(y_tangent, -1)

    assert grad_y.shape == mask.shape
    grad_y = jtu.tree_map(FwdJacobian, grad_y, mask)
    return y, grad_y, lapl_y


def general_jvp(
    vmapped_jvp: JvpFn,
    laplace_args: FwdLaplArgs,
):
    # General implementation. This will always materialize the full Jacobian.
    tangent = tree_concat(
        [
            extend_jacobians(*laplace_args.dense_jacobian, axis=JAC_DIM),
            tree_idx(laplace_args.laplacian, (None,)),
        ],
        axis=0,
    )
    y, y_tangent = vmapped_jvp(laplace_args.x, tangent)
    grad_y, lapl_y = tree_idx(y_tangent, slice(None, -1)), tree_idx(y_tangent, -1)

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

    def _jvp(primals: Arrays, tangents: Arrays) -> tuple[Array, Array]:
        return jax.jvp(merged_fwd, primals, tangents)  # type: ignore

    vmapped_jvp = jax.vmap(_jvp, in_axes=(None, 0), out_axes=(None, 0))
    vmapped_jvp.__name__ = merged_fwd.__name__

    def jvp(args: FwdLaplArgs, kwargs):
        # If any jacobian is dense, we just switch all jacobians to dense.
        if not args.all_jacobian_weak:
            return general_jvp(vmapped_jvp, args)

        # Special case for index operation
        if FunctionFlags.INDEXING in flags:
            return sparse_index_jvp(
                fwd,
                vmapped_jvp,
                args,
                extra_args,
                merge,
                index_static_args,
            )
        else:
            return sparse_reduction_jvp(
                vmapped_jvp,
                args,
                axes=in_axes,
                kwargs=kwargs,
                sparsity_threshold=sparsity_threshold,
            )

    return jvp
