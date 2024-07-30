import functools
import logging
from typing import Sequence, Type, TypeVar

import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .api import (
    IS_LEAF,
    JAC_DIM,
    Array,
    ArrayOrFwdLaplArray,
    Arrays,
    Axes,
    ExtraArgs,
    ForwardFn,
    FwdJacobian,
    FwdLaplArgs,
    FwdLaplArray,
    FwdLaplArrays,
    MergeFn,
    PyTree,
)

T = TypeVar('T')


def bound_axis(arr: np.ndarray, axis):
    if isinstance(axis, int):
        axis = (axis,)
    axis = np.array(axis, dtype=int)
    axis[axis < 0] += arr.ndim
    return tuple(axis)


def tree_shapes(tree: PyTree[Array]) -> list[tuple[int, ...]]:
    """
    Returns a list of shapes of the given tree.
    """
    leaves = jtu.tree_leaves(tree)
    return [l.shape for l in leaves]


def trace_of_product(mat1: Array, mat2: Array):
    """
    Computes the trace of the product of the given matrices.
    """
    # ij,ij->... is a faster way to compute the trace than tr(mat1@mat2)
    # since one can rewrite the trace as sum_ij mat1_ij * mat2_ij
    return jnp.einsum('...ij,...ij->...', mat1, mat2)


def get_reduced_jacobians(*jacs: FwdJacobian, idx: Array | np.ndarray | None):
    """
    Takes a sequence of jacobians and returns a sequence of
    jacobians where only the shared indices are kept.
    """
    if idx is None:
        data = [j.dense_array for j in jacs]
        data = extend_jacobians(*data, axis=JAC_DIM)
        data = [x.reshape(x.shape[0], -1) for x in data]
    else:
        data = [j.construct_jac_for(idx) for j in jacs]
        data = [x.reshape(len(idx), -1) for x in data]
    return data


def trace_jac_jacT(
    first: FwdJacobian, other: FwdJacobian, idx: Array | np.ndarray | None
):
    """
    Computes the trace of the product of the given jacobians.
    """
    x, y = get_reduced_jacobians(first, other, idx=idx)
    return trace_of_product(x, y)


def jac_jacT(first: FwdJacobian, other: FwdJacobian, idx: Array | np.ndarray | None):
    """
    Computes outer product of the given jacobians.
    """
    x, y = get_reduced_jacobians(first, other, idx=idx)
    return x @ y.T


def flat_wrap(fn: ForwardFn, *x: Array):
    """
    Wraps the given function such that it takes a flat array
    as input and returns a flat array as output. All inputs are expected
    to be concatenated.
    """
    _, x_unravel = jfu.ravel_pytree(x)

    def new_fn(flat_x: Array) -> Array:
        x = x_unravel(flat_x)
        return jfu.ravel_pytree(fn(*x))[0]  # type: ignore

    return new_fn


def array_wise_flat_wrap(fn: ForwardFn, *x: Array):
    """
    Wraps the given function such that it takes a flat arrays
    as input and returns a flat array as output.
    """
    unravels = [jfu.ravel_pytree(x_)[1] for x_ in x]

    def new_fn(*flat_x: Array) -> Array:
        x = [unravel(flat_x_) for unravel, flat_x_ in zip(unravels, flat_x)]
        return jfu.ravel_pytree(fn(*x))[0]  # type: ignore

    return new_fn


def broadcast_shapes_to_args(
    args: PyTree[ArrayOrFwdLaplArray], axes: PyTree[Axes]
) -> PyTree[Axes]:
    """
    Broadcasts the given axes to the given pytree of arrays.
    This is intended to replicate the broadcasting behavior of jax.vmap or jax.pmap.
    """

    def is_axes_def(x):
        if isinstance(x, tuple) and all(isinstance(x_, int) for x_ in x):
            return True
        if isinstance(x, int):
            return True
        if x is None:
            return True
        return False

    def canonicalize_axes(inp, x: ArrayOrFwdLaplArray):
        if not isinstance(x, (Array, FwdLaplArray)):
            x = jnp.array(x)
        if isinstance(inp, int):
            inp = (inp,)
        if inp is None:
            inp = tuple(range(x.ndim))
        n_dim = x.ndim
        inp = tuple(i if i >= 0 else i + n_dim for i in inp)
        assert (
            max(*inp, -1, -1) < n_dim
        ), f'axes {inp} out of bounds for array of shape {x.shape}'
        return inp

    flat_axes, tree_def = jtu.tree_flatten(axes, is_leaf=is_axes_def)
    leaves = tree_def.flatten_up_to(args)
    return tree_def.unflatten(
        [
            jtu.tree_map(lambda x: canonicalize_axes(a, x), l, is_leaf=IS_LEAF)
            for a, l in zip(flat_axes, leaves)
        ]
    )


def vmap_sequence(
    arr: Array, shape: tuple[int, ...], skip_axes: tuple[int, ...]
) -> list[int | None]:
    """
    Returns a list of indices that can be used to broadcast
    the given array to the given shape.
    """
    curr_dim = 0
    maps: list[int | None] = []
    active_dims = arr.ndim - len(skip_axes)
    if len(shape) > active_dims:
        maps = [None] * (len(shape) - active_dims)
        shape = shape[len(shape) - active_dims :]
    skipped = 0
    for i in range(arr.ndim):
        if i in skip_axes:
            skipped += 1
        elif arr.ndim < arr.ndim - i + curr_dim:
            maps.append(None)
        elif arr.shape[curr_dim + skipped] == 1:
            maps.append(None)
            curr_dim += 1
        else:
            maps.append(skipped)
            curr_dim += 1
    return maps


def remove_axes(arr: Array, axes: Array):
    """
    Removes the given axes from the given array by picking the
    first element along the given axes.
    """
    idx: list[slice | int] = []
    for i in range(arr.ndim):
        if i not in axes:
            idx.append(slice(None))
        else:
            idx.append(0)
    return arr[tuple(idx)]


def vmap_sequences(arrs: PyTree[Array], in_axes: Axes) -> list[PyTree[Axes]]:
    """
    Returns a list of pytrees of the same structure as arrs but with
    vmap in_axes as leaves such that all arrays can be broadcasted
    to the same shape.
    """
    in_axes = broadcast_shapes_to_args(arrs, in_axes)
    # TODO: This only works if the axes are at the back?!
    reduced_arrs = jtu.tree_map(remove_axes, arrs, in_axes)
    brdcast_shape = jnp.broadcast_shapes(*tree_shapes(reduced_arrs))
    flat_arrs, tree_def = jtu.tree_flatten(arrs)
    in_axes = tree_def.flatten_up_to(in_axes)
    maps = [
        vmap_sequence(arr, brdcast_shape, ax) for arr, ax in zip(flat_arrs, in_axes)
    ]
    keep_idx = []
    for i in range(len(brdcast_shape)):
        if any(m[i] is not None for m in maps):
            keep_idx.append(i)
    return [tree_def.unflatten([m[i] for m in maps]) for i in keep_idx]


def arg_squeeze_dims(arrs: PyTree[Array], in_axes: Axes) -> PyTree[Array]:
    in_axes = broadcast_shapes_to_args(arrs, in_axes)

    def _squeeze(arr, in_axes):
        squeeze_mask = np.array([i not in in_axes for i in range(arr.ndim)])
        squeeze_mask = np.logical_and(squeeze_mask, np.array(arr.shape) == 1)
        return jnp.squeeze(arr, tuple(np.where(squeeze_mask)[0]))

    return jtu.tree_map(_squeeze, arrs, in_axes)


def vmap_sequences_and_squeeze(arrs: PyTree[Array], in_axes: Axes = None):
    """
    Returns two things:
    - a sequence of pytrees of the same structure as arrs but with
      vmap in_axes as leaves such that all arrays can be broadcasted
      to the same shape. in_axes are kept.
    - the source arrays with all axes that are 1 in all arrays removed.
    """
    seqs = vmap_sequences(arrs, in_axes)
    sque_args = arg_squeeze_dims(arrs, in_axes)
    return seqs, sque_args


def add_vmap_jacobian_dim(args: FwdLaplArgs, in_axes: FwdLaplArgs):
    """
    Adds a new dimension to the given args.
    The new dimension is added to the jacobian of each array.
    """
    return FwdLaplArgs(
        tuple(
            FwdLaplArray(
                x=ax,  # type: ignore
                jacobian=(*(x + (x >= JAC_DIM) for x in ax), JAC_DIM),  # type: ignore
                laplacian=ax,  # type: ignore
            )
            for a, ax in zip(args.arrays, in_axes.arrays)
        )
    )


def split_args(
    args: tuple[ArrayOrFwdLaplArray, ...],
    in_axes: Axes,
    filter_type: Type = FwdLaplArray,
) -> tuple[FwdLaplArrays, Axes, ExtraArgs, Axes, MergeFn]:
    """
    Splits the given tree into two parts:
    - the first part contains all leaves of the given type
    - the second part contains all other leaves
    It returns the two parts as well as a function that assembles the
    two parts to the original tree.
    """
    brd_axes = broadcast_shapes_to_args(args, in_axes)
    leaves, tree_def = jtu.tree_flatten(
        args,
        is_leaf=lambda x: isinstance(x, filter_type)
        or not isinstance(x, (dict, list, tuple)),
    )
    flat_axes = tree_def.flatten_up_to(brd_axes)

    mask = [isinstance(l, filter_type) for l in leaves]
    args = tuple(l for l, m in zip(leaves, mask) if m)
    args_in_axes = tuple(a for a, m in zip(flat_axes, mask) if m)
    extra_args = tuple(l for l, m in zip(leaves, mask) if not m)
    extra_in_axes = tuple(a for a, m in zip(flat_axes, mask) if not m)

    def merge(args: Arrays, extra: ExtraArgs) -> Arrays:
        x_iter = iter(args)
        y_iter = iter(extra)
        return jtu.tree_unflatten(
            tree_def, [(next(x_iter) if m else next(y_iter)) for m in mask]
        )

    return (args, args_in_axes, extra_args, extra_in_axes, merge)


def ravel(pytree):
    """
    An implementation of jax.flatten_util.ravel_pytree that does not
    require the leaves to be jax.Array when unflattening.
    """
    leaves, tree_def = jtu.tree_flatten(pytree)
    shapes = [l.shape for l in leaves]
    flat = jnp.concatenate([l.ravel() for l in leaves])

    def unravel(arr):
        unravelled = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape, dtype=int)
            unravelled.append(arr[idx : idx + size].reshape(shape))
            idx += size
        return tree_def.unflatten(unravelled)

    return flat, unravel


def np_concatenate_brdcast(arrs, axis):
    """
    Concatenates the given arrays along the given axis.
    Before concatenation, the arrays are broadcasted to the same shape.

    Args:
        - arrs: sequence of arrays
        - axis: axis along which to concatenate
    Returns:
        - np.ndarray: np.ndarray where all arrays are broadcasted to the same shape
    """
    return np.concatenate(broadcast_except(arrs, axis), axis=axis)


def broadcast_except(arrs, axis):
    """
    Broadcasts all arrays to the same shape except for the specified axes.

    Args:
        - arrs: sequence of arrays
        - axes: tuple of integers specifying the axes to exclude from broadcasting
    Returns:
        - np.ndarray: sequence of arrays with the same shape except for the specified axes
    """
    if axis < 0:
        axis += jtu.tree_leaves(arrs)[0].ndim
    pre_shapes = [x.shape[:axis] for x in jtu.tree_leaves(arrs)]
    post_shapes = [x.shape[axis + 1 :] for x in jtu.tree_leaves(arrs)]
    max_pre = np.broadcast_shapes(*pre_shapes)
    max_post = np.broadcast_shapes(*post_shapes)

    def broadcast(a):
        broadcast = np.broadcast_to if isinstance(a, np.ndarray) else jnp.broadcast_to
        moveaxis = np.moveaxis if isinstance(a, np.ndarray) else jnp.moveaxis
        out = broadcast(moveaxis(a, axis, -1), (*max_pre, *max_post, a.shape[axis]))
        return moveaxis(out, -1, axis)  # type: ignore

    return jtu.tree_map(broadcast, arrs)


def extend_jacobians(*x: Array, axis):
    """
    Extends the given arrays to the same shape by appending zeros.
    """
    if len(x) == 1:
        return x
    if axis < 0:
        axis += jtu.tree_leaves(x)[0].ndim
    max_dim = max([a.shape[axis] for a in x])
    if all(a.shape[axis] == max_dim for a in x):
        return x
    result = []
    for a in x:
        a_shape = list(a.shape)
        if a_shape[axis] < max_dim:
            a_shape[axis] = max_dim - a.shape[axis]
            a = jnp.concatenate([a, jnp.zeros(a_shape, dtype=a.dtype)], axis=axis)
        result.append(a)
    return tuple(result)


def broadcast_dim(xs: Sequence[np.ndarray] | Sequence[Array], fill_value, axis):
    """
    Broadcasts all arrays to the same at the last dimension
    by repeating.
    """
    if axis < 0:
        axis += jtu.tree_leaves(xs)[0].ndim
    leaves, tree_def = jtu.tree_flatten(xs)
    max_dim = max([x.shape[axis] for x in leaves])
    return tree_def.unflatten(
        [
            jnp.concatenate(
                [
                    x,
                    np.full(
                        (
                            *x.shape[:axis],
                            max_dim - x.shape[axis],
                            *x.shape[axis + 1 :],
                        ),
                        fill_value,
                        dtype=x.dtype,
                    ),
                ],
                axis=axis,
            )
            for x in xs
        ]
    )


def compact_repeated_dims_except(arr: np.ndarray, axis):
    compact_axes_list = []
    dims = np.arange(arr.ndim)
    if axis is not None:
        axis = bound_axis(arr, axis)
        dims = np.setdiff1d(dims, axis)
    for d in dims:
        first_item = np.take(arr, [0], axis=d)
        if (first_item == arr).all():
            arr = first_item
            compact_axes_list.append(d)
    compact_axes = tuple(compact_axes_list)
    return arr, compact_axes


def get_jacobian_for_reduction(jacs: Sequence[FwdJacobian], axes):
    # The idea is to first rearrange the jacobians such that all batch dimensions
    # are reduced to one, all reduction dimensions are reduced to one and as last
    # we have the jacobian dimension. Then we can simply count for each output
    # on which inputs it depends. We can use this to construct a mapping from the
    # original jacobian to the reduced jacobian.
    jacs = tuple(jacs)
    axes = np.array(axes, dtype=int)
    axes[axes < 0] += jacs[0].data.ndim - 1

    if np.array(axes).ndim == 1:
        axes = axes[None]
    if len(axes) != len(jacs):
        axes = np.repeat(axes, len(jacs), axis=0)

    # Match shapes for masks
    masks = broadcast_except(tuple(map(lambda x: x.mask, jacs)), axis=JAC_DIM)
    # Compute a bunch of shapes and sizes
    reduction_shapes = tuple(
        tuple(np.array(m.shape[1:])[a]) for a, m in zip(axes, masks)
    )
    assert all(len(reduction_shapes[0]) == len(s) for s in reduction_shapes)
    reduction_size = np.prod(reduction_shapes[0], dtype=int)

    jac_reduced_axes = tuple(
        (*[x + int(x >= JAC_DIM) for x in a], JAC_DIM)
        for a in axes  # the first dim is the same for all arrays
    )
    kept_axes = tuple(
        np.setdiff1d(np.arange(masks[0].ndim), a) for a in jac_reduced_axes
    )
    kept_shapes = tuple(tuple(np.array(m.shape)[a]) for a, m in zip(kept_axes, masks))
    assert all(kept_shapes[0] == s for s in kept_shapes)
    kept_shape = kept_shapes[0]
    kept_size = np.prod(kept_shape, dtype=int)

    inv_orders = tuple(
        tuple(np.argsort((*kept_axes[i], *jac_reduced_axes[i])))
        for i in range(len(jacs))
    )

    # Let's rearrange masks and data such that all batch dimensions are reduced
    # to one, all reduction dimensions are reduced to one and as last we have the
    # jacobian dimension.
    def rearrange(
        mask,
        kept_axes,
        jac_reduced_axes,
    ):
        transpose = np.transpose if isinstance(mask, np.ndarray) else jnp.transpose
        return transpose(mask, (*kept_axes, *jac_reduced_axes)).reshape(
            kept_size, reduction_size, -1
        )

    masks = jtu.tree_map(rearrange, masks, kept_axes, jac_reduced_axes)

    # Determine for each element the outputs.
    mask = np.concatenate(masks, axis=-1)
    out_mask_list = [np.unique(m) for m in mask]
    out_mask_list = [m[m != -1] for m in out_mask_list]
    max_unique = max([m.size for m in out_mask_list])

    # Here we extend each mask to the same size by appending -1.
    out_mask = np.stack(
        [
            np.concatenate([m, np.full(max_unique - m.size, -1, dtype=np.int32)])
            for m in out_mask_list
        ]
    )

    # Let's reconstruct the original order for the output mask
    out_masks = tuple(
        np.transpose(
            out_mask.reshape(*kept_shape, *([1] * len(reduction_shapes[0])), -1),
            inv_order,
        )
        for inv_order in inv_orders
    )

    # Materialize the needed jacobian
    jacobians = tuple(
        jac.materialize_for_idx(
            jac.get_index_mask(mask),
            max_idx=max_unique,
        )
        for jac, mask in zip(jacs, out_masks)
    )
    # Remove all contracted dimensions again
    out_mask = out_masks[0].reshape(-1, *kept_shape)
    return jacobians, out_mask


def add_jacobians(jac1: Array, jac2: Array):
    """
    Adds two dense jacobians.
    """
    jac1, jac2 = extend_jacobians(jac1, jac2, axis=JAC_DIM)
    return jac1 + jac2


def extract_jacobian_mask(arrays: Sequence[ArrayOrFwdLaplArray]):
    indices = []
    for arr in arrays:
        if isinstance(arr, FwdLaplArray):
            indices.append(arr.jacobian.x0_idx)

    def merge(arrs: ArrayOrFwdLaplArray):
        idx_iter = iter(indices)
        return [
            arr._replace(jacobian=arr.jacobian._replace(x0_idx=next(idx_iter)))
            if isinstance(arr, FwdLaplArray)
            else arr
            for arr in arrs
        ]

    return merge


def broadcast_mask_to_jacobian(mask: PyTree[np.ndarray], jacobian: PyTree[Array]):
    """
    Broadcasts the given mask to the given jacobian.
    """

    def broadcast(m: np.ndarray, j: Array):
        assert m.shape[JAC_DIM] == j.shape[JAC_DIM]
        target_shape = list(j.shape)
        del target_shape[JAC_DIM]
        target_shape = tuple(target_shape)

        @functools.partial(jax.vmap, in_axes=JAC_DIM, out_axes=JAC_DIM)
        def brdcast(x):
            return jnp.broadcast_to(x, target_shape)

        with jax.ensure_compile_time_eval():
            return np.asarray(brdcast(m), dtype=m.dtype)

    return jtu.tree_map(broadcast, mask, jacobian)


class LoggingPrefix(logging.Formatter):
    prefix: str
    _old_handlers = None

    def __init__(self, prefix: str):
        self.prefix = prefix
        super().__init__()

    def format(self, record):
        record.msg = f'{record.levelname}:[folx]{self.prefix} - {record.msg}'
        return super().format(record)

    def __enter__(self):
        logger = logging.getLogger()
        self._old_handlers = logger.handlers
        myHandler = logging.StreamHandler()
        myHandler.setFormatter(self)
        logger.handlers = [myHandler]

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logging.error(
                'Exception occurred', exc_info=(exc_type, exc_value, traceback)
            )
        logger = logging.getLogger()
        logger.handlers = self._old_handlers
