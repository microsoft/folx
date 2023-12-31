import functools
from typing import ParamSpec, TypeVar

import jax

from .api import (
    IS_LPL_ARR,
    Array,
    ArrayOrFwdLaplArray,
    Axes,
    CustomTraceJacHessianJac,
    ExtraArgs,
    ForwardFn,
    ForwardLaplacian,
    ForwardLaplacianFns,
    FunctionFlags,
    FwdLaplArgs,
    FwdLaplArray,
    MergeFn,
    PyTree,
)
from .hessian import get_jacobian_hessian_jacobian_trace
from .jvp import get_jvp_function
from .tree_utils import tree_add
from .utils import split_args

R = TypeVar("R", bound=PyTree[Array])
P = ParamSpec("P")


def construct_fwd_laplacian_functions(
    fwd: ForwardFn,
    flags: FunctionFlags,
    in_axes: Axes,
    extra_args: ExtraArgs,
    arg_axes: Axes,
    extra_in_axes: Axes,
    merge: MergeFn,
    index_static_args: tuple | slice | None,
    sparsity_threshold: int,
    custom_jac_hessian_jac: CustomTraceJacHessianJac | None
):
    def merged_fwd(*args: Array):
        return fwd(*merge(args, extra_args))

    merged_fwd.__name__ = fwd.__name__
    return ForwardLaplacianFns(
        merged_fwd,
        get_jvp_function(
            fwd, flags, in_axes, extra_args, merge, index_static_args, sparsity_threshold
        ),
        get_jacobian_hessian_jacobian_trace(
            fwd=fwd,
            flags=flags,
            custom_jac_hessian_jac=custom_jac_hessian_jac,
            extra_args=extra_args,
            in_axes=arg_axes,
            extra_in_axes=extra_in_axes,
            merge=merge
        ),
    )


def wrap_forward_laplacian(
    fn: ForwardFn,
    in_axes: Axes = None,
    flags: FunctionFlags = FunctionFlags.GENERAL,
    name: str | None = None,
    index_static_args: tuple | slice | None = None,
    custom_jac_hessian_jac: CustomTraceJacHessianJac | None = None
) -> ForwardLaplacian:
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
            custom_jac_hessian_jac=custom_jac_hessian_jac
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


def warp_without_fwd_laplacian(fn) -> ForwardLaplacian:
    """
    Decorator that removes the Laplacian state from the arguments of a function.
    """
    def wrapped(*args, sparsity_threshold: int, **kwargs):
        args, kwargs = jax.tree_util.tree_map(
            lambda a: (a.x if isinstance(a, FwdLaplArray) else a),
            (args, kwargs),
            is_leaf=IS_LPL_ARR,
        )
        return fn(*args, **kwargs)

    return wrapped
