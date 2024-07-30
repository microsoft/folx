import jax
import jax.numpy as jnp

from folx.ad import is_tree_complex
from folx.vmap import batched_vmap

from .api import JAC_DIM, Array, ExtraArgs, FwdLaplArgs, MergeFn
from .utils import get_reduced_jacobians, trace_jac_jacT, trace_of_product


def slogdet_jac_hessian_jac(
    args: FwdLaplArgs,
    extra_args: ExtraArgs,
    merge: MergeFn,
    materialize_idx: Array | None,
):
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
        # We can do better and compute the trace more efficiently.
        A_inv_J = jnp.einsum('ij,jdk->idk', A_inv, J)
        trace = -trace_of_product(
            jnp.transpose(A_inv_J, (1, 0, 2)).reshape(-1, x0_dim),
            A_inv_J.reshape(-1, x0_dim),
        )
        return jnp.zeros((), dtype=trace.dtype), trace

    A_inv = A_inv.reshape(-1, *A.shape[-2:])
    J = J.reshape(-1, *J.shape[-3:])

    # We can either use vmap or scan. Scan is slightly slower but uses less memory.
    # Here we assume that we will in general encounter larger determinants rather than many.
    signs, flat_out = batched_vmap(elementwise, 1)(A_inv, J)
    sign_out, log_abs_out = signs.reshape(leading_dims), flat_out.reshape(leading_dims)

    if is_tree_complex(A):
        # this is not the real Tr(JHJ^T) but a cached value we use later to compute the Tr(JHJ^T)
        return log_abs_out, log_abs_out.real
    return sign_out, log_abs_out.real


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
