import jax
import jax.numpy as jnp

from folx.ad import is_tree_complex

from .api import Array, ExtraArgs, FwdLaplArgs, MergeFn
from .utils import get_reduced_jacobians, trace_jac_jacT


def complex_abs_jac_hessian_jac(
    args: FwdLaplArgs,
    extra_args: ExtraArgs,
    merge: MergeFn,
    materialize_idx: Array | None,
):
    # The hessian of jnp.abs seems to be numerically unstable.
    # Here we implement a custom rule based on
    # abs(x) = sqrt(x.real^2 + x.imag^2)
    assert len(args.x) == 1
    # This function is applied elementwise
    assert args.x[0].shape == ()

    # For real numbers the Hessian is 0.
    if not is_tree_complex(args.x):
        return jnp.zeros(())

    x, J = args.x[0], args.jacobian[0].data
    y, J_abs = jnp.abs(x), jnp.abs(J)

    x_J = x.real * J.real + x.imag * J.imag

    return jnp.vdot(J_abs, J_abs) / y - jnp.vdot(x_J, x_J) / y**3


def div_jac_hessian_jac(
    args: FwdLaplArgs,
    extra_args: ExtraArgs,
    merge: MergeFn,
    materialize_idx: Array | None,
):
    total_args = merge(args.x, extra_args)  # type: ignore
    assert len(total_args) == 2, 'Div requires two arguments'
    x, y = total_args
    assert x.shape == y.shape == (), 'Div requires two scalars'

    if len(args.x) == 2:
        lhs_grad, rhs_grad = get_reduced_jacobians(*args.jacobian, idx=materialize_idx)
        JJ_rr = rhs_grad.T @ rhs_grad
        JJ_lr = lhs_grad.T @ rhs_grad

        ry2 = jax.lax.integer_pow(y, -2)
        H_lr = -2 * ry2
        H_rr = 2 * x * (ry2 / y)
        return H_lr * JJ_lr + H_rr * JJ_rr
    elif len(args.x) == 1:
        # We know that it must be the denominator since the Hessian would be 0 otherwise.
        J_den = args.jacobian[0]
        H = 2 * x * jax.lax.integer_pow(y, -3)
        return trace_jac_jacT(J_den, J_den, materialize_idx) * H
