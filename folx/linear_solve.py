"""Forward laplacian of linear solves.

``A^-1`` appears in the derivatives of both ``A^-1 b`` and ``A^-1``, so a single
inverse serves the jacobian and the laplacian of either.
"""

from typing import Any

import jax.numpy as jnp

from .api import JAC_DIM, Array, ArrayOrFwdLaplArray, FwdJacobian, FwdLaplArray
from .utils import extend_jacobians


def _operand(v: ArrayOrFwdLaplArray):
    """Value, dense jacobian and laplacian of an operand.

    Args:
        v: Either a constant array or a forward laplacian array.

    Returns:
        Tuple of the value and, if the operand varies, its dense jacobian and its
        laplacian, else None for both.
    """
    if isinstance(v, FwdLaplArray):
        return v.x, v.jacobian.dense_array, v.laplacian
    return v, None, None


def _sub(total: Array | None, term: Array):
    """Subtracts a term from a possibly absent total."""
    return -term if total is None else total - term


def _align(jac: Array, shape: tuple[int, ...]):
    """Aligns a jacobian's trailing axes with a batch shape.

    The tangent axis comes first, so an operand with fewer batch axes than the
    output needs them inserted rather than broadcast from the right.

    Args:
        jac: Jacobian of shape ``(tangents, ...)``.
        shape: Shape the axes after the tangent axis should broadcast against.

    Returns:
        The jacobian with singleton axes inserted after the tangent axis.
    """
    missing = len(shape) - (jac.ndim - 1)
    if missing <= 0:
        return jac
    return jac.reshape(jac.shape[JAC_DIM], *(1,) * missing, *jac.shape[1:])


def solve_wrapper(
    x: tuple[ArrayOrFwdLaplArray, ArrayOrFwdLaplArray],
    kwargs: dict[str, Any],
    sparsity_threshold: int,
):
    """Forward laplacian of ``y = A^-1 b``.

    Differentiating ``A y = b`` once and twice gives

        A d_c y     = d_c b - d_c A y
        A Delta y   = Delta b - Delta A y - 2 sum_c d_c A d_c y

    so all outputs are solves against the same matrix, which share its
    factorization, plus one matrix product per tangent row. Either operand may be
    constant.

    Args:
        x: Tuple of the matrix, shape ``(..., n, n)``, and the right hand side,
            shape ``(..., n)`` or ``(..., n, m)``.
        kwargs: Unused.
        sparsity_threshold: Unused; the solution depends on every coordinate its
            operands depend on.

    Returns:
        The FwdLaplArray of ``y``, or the plain solution if neither operand varies.
    """
    A, b = x
    A_x, A_jac, A_lapl = _operand(A)
    b_x, b_jac, b_lapl = _operand(b)
    if A_jac is None and b_jac is None:
        return jnp.linalg.solve(A_x, b_x)
    # jnp.linalg.solve reads a one dimensional right hand side as a single vector
    # and anything else as a matrix; a trailing axis makes every case a matrix,
    # which also lets solve broadcast over the leading tangent axis
    vector = b_x.ndim == 1
    if vector:
        b_x = b_x[..., None]
        b_jac = None if b_jac is None else b_jac[..., None]
        b_lapl = None if b_lapl is None else b_lapl[..., None]
    if A_jac is not None and b_jac is not None:
        A_jac, b_jac = extend_jacobians(A_jac, b_jac, axis=JAC_DIM)

    y = jnp.linalg.solve(A_x, b_x)
    jac_rhs = None if b_jac is None else _align(b_jac, y.shape)
    lapl_rhs = b_lapl
    if A_jac is not None:
        jac_rhs = _sub(jac_rhs, jnp.einsum('k...il,...lm->k...im', A_jac, y))
        lapl_rhs = _sub(lapl_rhs, jnp.einsum('...il,...lm->...im', A_lapl, y))
    jacobian = jnp.linalg.solve(A_x, jac_rhs)
    if A_jac is not None:
        lapl_rhs = _sub(
            lapl_rhs, 2 * jnp.einsum('k...il,k...lm->...im', A_jac, jacobian)
        )
    laplacian = jnp.linalg.solve(A_x, lapl_rhs)
    if vector:
        y, jacobian, laplacian = y[..., 0], jacobian[..., 0], laplacian[..., 0]
    return FwdLaplArray(y, FwdJacobian.from_dense(jacobian), laplacian)


def inv_wrapper(
    x: tuple[ArrayOrFwdLaplArray],
    kwargs: dict[str, Any],
    sparsity_threshold: int,
):
    """Forward laplacian of ``C = A^-1``.

    Differentiating ``A C = I`` once and twice gives, with ``M_c = C dA/dx_c``,

        d_c C       = -M_c C
        Delta C     = (2 sum_c M_c M_c - C Delta A) C

    so the inverse and the products ``M_c`` are shared by both outputs.

    Args:
        x: Tuple holding the matrix, shape ``(..., n, n)``.
        kwargs: Unused.
        sparsity_threshold: Unused; an inverse depends on every coordinate its
            matrix depends on.

    Returns:
        The FwdLaplArray of ``A^-1``.
    """
    A = x[0]
    if not isinstance(A, FwdLaplArray):
        return jnp.linalg.inv(A)
    C = jnp.linalg.inv(A.x)
    M = jnp.einsum('...il,k...lj->k...ij', C, A.jacobian.dense_array)
    jacobian = -jnp.einsum('k...il,...lj->k...ij', M, C)
    inner = 2 * jnp.einsum('k...il,k...lj->...ij', M, M)
    inner -= jnp.einsum('...il,...lj->...ij', C, A.laplacian)
    laplacian = jnp.einsum('...il,...lj->...ij', inner, C)
    return FwdLaplArray(C, FwdJacobian.from_dense(jacobian), laplacian)
