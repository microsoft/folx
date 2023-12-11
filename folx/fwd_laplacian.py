import functools
from typing import ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .api import (IS_LPL_ARR, Array, ArrayOrFwdLaplArray, Axes, ExtraArgs,
                  ForwardFn, ForwardLaplacianFns, FunctionFlags, FwdJacobian,
                  FwdLaplArgs, FwdLaplArray, MergeFn)
from .hessian import get_jacobian_hessian_jacobian_trace
from .jvp import get_jvp_function
from .tree_utils import tree_add
from .types import PyTree
from .utils import ravel, split_args

R = TypeVar("R", bound=PyTree[Array])
P = ParamSpec("P")


def construct_fwd_laplacian_functions(
    fwd: ForwardFn,
    fn_flags: FunctionFlags,
    in_axes: Axes,
    extra_args: ExtraArgs,
    arg_axes: Axes,
    extra_in_axes: Axes,
    merge: MergeFn,
    index_static_args: tuple | slice | None,
    sparsity_threshold: int,
):
    def merged_fwd(*args: Array):
        return fwd(*merge(args, extra_args))

    merged_fwd.__name__ = fwd.__name__
    return ForwardLaplacianFns(
        merged_fwd,
        get_jvp_function(
            fwd, fn_flags, in_axes, extra_args, merge, index_static_args, sparsity_threshold
        ),
        get_jacobian_hessian_jacobian_trace(
            fwd, fn_flags, extra_args, arg_axes, extra_in_axes, merge
        ),
    )


def add_forward_laplacian(
    fn: ForwardFn,
    in_axes: Axes = None,
    flags: FunctionFlags = FunctionFlags.GENERAL,
    name: str | None = None,
    index_static_args: tuple | slice | None = None,
):
    """
    Add forward Laplacian functionality to a function.

    Args:
        - fn: function to add forward Laplacian functionality to
        - in_axes: axes that are needed for the computation
        - flags: flags that specify the type of function
    """

    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    def new_fn(
        *args: ArrayOrFwdLaplArray, sparsity_threshold: int, **kwargs
    ) -> PyTree[ArrayOrFwdLaplArray]:
        # split arguments into ForwardLaplacianArrays and other arrays
        lapl_args, lapl_axes, extra, extra_axes, merge = split_args(args, in_axes)

        # construct operation
        partial_fn = functools.partial(fn, **kwargs)
        setattr(partial_fn, "__name__", name or getattr(fn, "__name__", "partial"))
        lapl_fns = construct_fwd_laplacian_functions(
            partial_fn,
            flags,
            in_axes=in_axes,
            extra_args=extra,
            arg_axes=lapl_axes,
            extra_in_axes=extra_axes,
            merge=merge,
            index_static_args=index_static_args,
            sparsity_threshold=sparsity_threshold,
        )
        # If we have no args just do regular forward pass
        if len(lapl_args) == 0:
            return lapl_fns.forward(*merge((), extra))

        # Actually update Laplacian state
        laplace_args = FwdLaplArgs(lapl_args)
        y, grad_y, lapl_y = lapl_fns.jvp(laplace_args, kwargs)
        lapl_y = tree_add(lapl_y, lapl_fns.jac_hessian_jac_trace(laplace_args, sparsity_threshold))
        return jax.tree_util.tree_map(FwdLaplArray, y, grad_y, lapl_y)

    return new_fn


def init_forward_laplacian_state(*x: PyTree[Array], sparsity: bool) -> PyTree[FwdLaplArray]:
    """
    Initialize forward Laplacian state from a PyTree of arrays.
    """
    x_flat, unravel = ravel(x)
    jac = jtu.tree_map(jnp.ones_like, x)
    jac_idx = unravel(np.arange(x_flat.shape[0]))
    if sparsity:
        jac = jtu.tree_map(lambda j, i: FwdJacobian(j[None], np.array(i)[None]), jac, jac_idx)
    else:
        jac = jax.vmap(unravel)(jnp.eye(len(x_flat)))
        jac = jtu.tree_map(FwdJacobian.from_dense, jac)
    lapl_x = jtu.tree_map(jnp.zeros_like, x)
    return jtu.tree_map(FwdLaplArray, x, jac, lapl_x)


def non_lapl_call(fn):
    """
    Decorator that removes the Laplacian state from the arguments of a function.
    """

    def wrapped(*args, sparsity_threshold=None, **kwargs):
        args, kwargs = jax.tree_util.tree_map(
            lambda a: (a.x if isinstance(a, FwdLaplArray) else a),
            (args, kwargs),
            is_leaf=IS_LPL_ARR,
        )
        return fn(*args, **kwargs)

    return wrapped
