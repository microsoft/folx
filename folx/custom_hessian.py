import jax
import jax.numpy as jnp

from .api import Array, ExtraArgs, FwdLaplArgs, MergeFn, JAC_DIM
from .utils import trace_of_product


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
        # Naive implementation
        # @functools.partial(jax.vmap, in_axes=(-1, None), out_axes=-1)
        # @functools.partial(jax.vmap, in_axes=(None, -1), out_axes=-1)
        # def inner(v1, v2):
        #     A_inv_v = A_inv@v1
        #     v_A_inv = v2.T@A_inv.T
        #     return -v_A_inv.reshape(-1)@A_inv_v.reshape(-1)
        # vHv = inner(J, J)
        # trace = jnp.trace(vHv)

        # We can do better and compute the trace more efficiently.
        A_inv_J = jnp.einsum('ij,jdk->idk', A_inv, J)
        trace = -trace_of_product(
            jnp.transpose(A_inv_J, (1, 0, 2)).reshape(-1, x0_dim),
            A_inv_J.reshape(-1, x0_dim),
        )
        return jnp.zeros(()), trace

    A_inv = A_inv.reshape(-1, *A.shape[-2:])
    J = J.reshape(-1, *J.shape[-3:])

    # We can either use vmap or scan. Scan is slightly slower but uses less memory.
    # Here we assume that we will in general encounter larger determinants rather than many.
    # signs, flat_out = jax.vmap(elementwise)(A_inv, J)
    def scan_wrapper(_, x):
        return None, elementwise(*x)

    signs, flat_out = jax.lax.scan(scan_wrapper, None, (A_inv, J))[1]
    return signs.reshape(leading_dims), flat_out.reshape(leading_dims)
