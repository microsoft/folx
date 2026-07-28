"""Forward laplacian of ``slogdet``.

The jacobian and the hessian trace of ``log|det A|`` both follow from the
inverse of ``A``. If the jacobian of ``A`` is sparse, the hessian trace also
decomposes over the input coordinates, which ``sparse_slogdet`` exploits; the
dense contraction is in ``dense_slogdet``.
"""

from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .ad import is_tree_complex
from .api import (
    JAC_DIM,
    Array,
    ArrayOrFwdLaplArray,
    FwdJacobian,
    FwdLaplArray,
)
from .utils import mark_varying_like


@jax.custom_jvp
def slogdet(x):
    # We only need this custom slog det to avoid a jax bug
    # https://github.com/google/jax/issues/17379
    # We explictily decompose this here as newer version will return
    # SlogDetResult which is a NamedTuple and does not combine nicely with regular tuples.
    sign, logdet = jnp.linalg.slogdet(x)
    return sign, logdet


def slogdet_jvp(primals, tangents):
    # we know that the input will be a single tensor where the last two dims will be reduced
    # So, instead of using the JVP from JAX, we compute the jacobian explicitly via backprop
    # and then compute the dot product with the tangent.
    primals, tangents = primals[0], tangents[0]
    batch_shape = primals.shape[:-2]  # the last two will be reduced
    tangents = tangents.reshape(-1, *tangents.shape[-2:])
    primals = primals.reshape(-1, *primals.shape[-2:])

    sign, logdet = jnp.linalg.slogdet(primals)
    y = sign, logdet

    jacobians = jnp.linalg.inv(primals)

    def custom_jvp(jacobian, tangent, sign):
        jac_dot_tangent = jnp.vdot(jacobian.T.conj(), tangent)
        if jac_dot_tangent.dtype in (jnp.complex64, jnp.complex128):
            # this is not the real jvp but a cached value to ease the Tr(JHJ^T) computation
            sign_jvp = jac_dot_tangent
            log_det_jvp = jac_dot_tangent.real
        else:
            sign_jvp = mark_varying_like(
                jnp.zeros((), dtype=jac_dot_tangent.dtype), jac_dot_tangent
            )
            log_det_jvp = jac_dot_tangent
        return (sign_jvp, log_det_jvp)

    y_tangent = jax.vmap(custom_jvp)(jacobians, tangents, sign)

    y, y_tangent = jtu.tree_map(lambda x: x.reshape(*batch_shape), (y, y_tangent))
    return y, y_tangent


slogdet.defjvp(slogdet_jvp)


_MAX_MASK_ENTRIES = 2**24  # bound for the compile time mask analysis


def _transposed(A: FwdLaplArray) -> FwdLaplArray:
    """Transposes the last two axes of a matrix valued forward laplacian array.

    Args:
        A: Forward laplacian array of a matrix, shape ``(..., n, m)``.

    Returns:
        The forward laplacian array of ``A^T``.
    """
    jac = A.jacobian
    x0_idx = None if jac.x0_idx is None else np.swapaxes(jac.x0_idx, -1, -2)
    return FwdLaplArray(
        jnp.swapaxes(A.x, -1, -2),
        FwdJacobian(jnp.swapaxes(jac.data, -1, -2), x0_idx),
        jnp.swapaxes(A.laplacian, -1, -2),
    )


def _entry_mask(A: FwdLaplArray, transposed: bool):
    """Per-entry coordinate mask of the jacobian, shared across batch positions.

    Args:
        A: Forward laplacian array of the matrix, shape ``(..., n, m)``.
        transposed: Whether to describe ``A^T`` instead of ``A``.

    Returns:
        Mask of shape ``(tangents, n, m)`` mapping every ``(tangent, row, column)``
        entry to an input coordinate, ``-1`` for unused entries. None if the
        jacobian is dense, the mask differs between batch positions or it is too
        large to analyze at trace time.
    """
    idx, shape = A.jacobian.x0_idx, A.x.shape
    if idx is None:
        return None
    if transposed:
        if idx.ndim != len(shape) + 1:
            return None
        idx = np.swapaxes(idx, -1, -2)
        shape = (*shape[:-2], shape[-1], shape[-2])
    k = A.jacobian.data.shape[JAC_DIM]
    if k * int(np.prod(shape, dtype=int)) > _MAX_MASK_ENTRIES:
        return None
    mask = np.broadcast_to(idx, (k, *shape))
    mask = np.moveaxis(mask, JAC_DIM, -3).reshape(-1, k, *shape[-2:])
    if not (mask == mask[:1]).all():
        return None
    return np.ascontiguousarray(mask[0])


def _slots(group: np.ndarray, n_groups: int):
    """Position of every element within its group and the group sizes.

    Args:
        group: Non-decreasing group index of every element.
        n_groups: Number of groups.

    Returns:
        Tuple of the within-group position of every element and the group sizes.
    """
    sizes = np.bincount(group, minlength=n_groups)
    return np.arange(group.size) - (np.cumsum(sizes) - sizes)[group], sizes


def _factor_tables(mask: np.ndarray, per_entry: bool):
    """Decomposes every ``dA/dx_c`` into rank one factors ``e_i w^T``.

    The factors are the rows of ``dA/dx_c``, restricted to the entries it is
    supported on: one factor per touched row, of the width needed for that row.
    With ``per_entry`` every entry becomes its own factor of width one instead,
    which is cheaper when the touched rows differ a lot in width, since the tables
    are padded to the widest one.

    Args:
        mask: Mask of shape ``(tangents, n, m)`` mapping every ``(tangent, row,
            column)`` entry to an input coordinate, ``-1`` for unused entries.
        per_entry: Whether to use one factor per entry instead of per row.

    Returns:
        ``(cost, (take, cols, rows, coords, per_entry))``: for the ``l``-th entry
        of the ``s``-th factor of ``coords[c]``, ``take[c, s, l]`` addresses it in
        the flattened ``(tangent, row, column)`` jacobian and ``cols[c, s, l]`` is
        its column; ``rows[c, s]`` is the row of the factor. Unused table entries
        address an appended zero. ``cost`` estimates the multiplications of the
        resulting contraction. None if the tables would be too large.
    """
    k, n, m = mask.shape
    (pos,) = np.nonzero(mask.reshape(-1) >= 0)
    coords, group = np.unique(mask.reshape(-1)[pos], return_inverse=True)
    group = group.reshape(-1)
    row, col = np.divmod(pos % (n * m), m)
    # sort the entries by coordinate, row and column so all groups are contiguous
    order = np.lexsort((col, row, group))
    group, pos, row, col = group[order], pos[order], row[order], col[order]
    key = group * (n * m) + row * m + (col if per_entry else 0)
    factors, factor_of = np.unique(key, return_inverse=True)
    slot, per_coord = _slots(factors // (n * m), coords.size)
    entry, per_factor = _slots(factor_of.reshape(-1), factors.size)
    rank, width = int(per_coord.max()), int(per_factor.max())
    if coords.size * rank * width > _MAX_MASK_ENTRIES:
        return None
    take = np.full((coords.size, rank, width), k * n * m, dtype=int)
    cols = np.zeros((coords.size, rank, width), dtype=int)
    rows = np.zeros((coords.size, rank), dtype=int)
    take[group, slot[factor_of], entry] = pos
    cols[group, slot[factor_of], entry] = col
    rows[factors // (n * m), slot] = factors % (n * m) // m
    # the W_c contraction and the gather of A^-1's entries it contracts with; entry
    # factors contract nothing because their entries are summed beforehand
    cost = coords.size * rank * rank * (1 if per_entry else width)
    return cost, (take, cols, rows, coords, per_entry)


def _sparse_slogdet(A: FwdLaplArray, tables):
    """Forward laplacian of log|det A| from the rank one factors of ``dA/dx_c``.

    Args:
        A: Forward laplacian array of the matrix, shape (..., n, n).
        tables: Static tables as returned by ``_factor_tables``.

    Returns:
        Tuple of sign and the FwdLaplArray of log|det A|.
    """
    take, cols, rows, coords, per_entry = tables
    data = A.jacobian.data
    shape = np.broadcast_shapes(data.shape[1:], A.x.shape[-2:])
    J = jnp.moveaxis(jnp.broadcast_to(data, (data.shape[JAC_DIM], *shape)), JAC_DIM, -3)
    J = J.reshape(*J.shape[:-3], -1)
    # the appended zero absorbs the unused table entries
    J = jnp.concatenate([J, jnp.zeros((*J.shape[:-1], 1), J.dtype)], axis=-1)
    A_inv = jnp.linalg.inv(A.x)
    vals = J[..., take]
    # W[c, s, t] = sum_l dA[i_s, j_l]/dx_c A^-1[j_l, i_t]
    if per_entry:
        # all entries of a factor share its column, so they can be summed first
        W = vals.sum(-1)[..., None] * A_inv[..., cols[:, :, :1], rows[:, None, :]]
    else:
        W = jnp.einsum(
            '...csl,...cslt->...cst',
            vals,
            A_inv[..., cols[..., None], rows[:, None, None, :]],
        )
    return _pack_slogdet(A, A_inv, W, coords)


def _pack_slogdet(A: FwdLaplArray, A_inv: Array, W: Array, coords: np.ndarray):
    """Assembles log|det A| from the per-coordinate matrices ``W_c``.

    With ``dA/dx_c = sum_s u_s v_s^T`` and ``W_c = V_c^T A^-1 U_c``,

        d_c log|det A|      = tr(W_c)
        Delta log|det A|    = tr(A^-1 Delta A) - sum_c sum_st W_c[s, t] W_c[t, s]

    Args:
        A: Forward laplacian array of the matrix, shape ``(..., n, n)``.
        A_inv: The inverse of ``A``.
        W: The matrices ``W_c``, shape ``(..., coords, rank, rank)``.
        coords: Input coordinate of every ``W_c``.

    Returns:
        Tuple of sign and the FwdLaplArray of log|det A|.
    """
    sign, logdet = slogdet(A.x)
    jacobian = jnp.einsum('...css->...c', W)
    laplacian = jnp.einsum('...ji,...ij->...', A_inv, A.laplacian)
    laplacian -= jnp.einsum('...cst,...cts->...', W, W)
    # the output is materialized because a determinant generally depends on all
    # coordinates it is sparse in
    jacobian = jnp.moveaxis(jacobian, -1, JAC_DIM)
    out_idx = np.broadcast_to(
        coords.reshape(-1, *(1,) * (jacobian.ndim - 1)), jacobian.shape
    )
    jacobian = FwdJacobian.from_dense(FwdJacobian(jacobian, out_idx).dense_array)
    return sign, FwdLaplArray(logdet, jacobian, laplacian)


def sparse_slogdet(A: FwdLaplArray):
    """Forward laplacian of log|det A| exploiting the jacobian's sparsity.

    Decomposes every ``dA/dx_c`` into rank one factors, which reduces both outputs
    to the small matrices ``W_c = V_c^T A^-1 U_c`` (see ``_pack_slogdet``). The
    factors are rows or single entries of ``dA/dx_c`` (see ``_factor_tables``), of
    either ``A`` or ``A^T``, so coordinates touching few rows, few columns or few
    entries all end up cheap. All four decompositions are estimated and the
    cheapest is used, provided it beats contracting the dense jacobian.

    Args:
        A: Forward laplacian array of the matrix, shape (..., n, n).

    Returns:
        Tuple of sign and the FwdLaplArray of log|det A|, or None for complex
        matrices, jacobians without exploitable sparsity, or masks for which
        contracting the dense jacobian is cheaper.
    """
    if is_tree_complex(A.x):
        return None
    best = None
    for transposed in (False, True):
        mask = _entry_mask(A, transposed)
        if mask is None or not (mask >= 0).any():
            continue
        # entry factors are invariant under transposition
        for per_entry in (False,) if transposed else (False, True):
            plan = _factor_tables(mask, per_entry)
            if plan is not None and (best is None or plan[0] < best[0]):
                best = (*plan, transposed)
    if best is None:
        return None
    cost, tables, transposed = best
    n, m = A.x.shape[-2:]
    tangents = A.jacobian.max_n + 1
    # the dense contraction runs n m multiplications per column of A^-1 dA/dx_c
    # plus n m for its trace contribution, over an n m intermediate per tangent
    # row. ``cost`` counts both the factorization's multiplications and the
    # entries it gathers, so it needs a margin over the former to pay off and
    # must not exceed the latter, or a flop win would cost memory instead.
    if 4 * cost > tangents * n * m * (n + 1) or cost > 2 * tangents * n * m:
        return None
    return _sparse_slogdet(_transposed(A) if transposed else A, tables)


def dense_slogdet(A: FwdLaplArray):
    """Forward laplacian of slogdet for a dense jacobian.

    Both outputs follow from the single product ``M_c = A^-1 dA/dx_c``,

        d_c log det A       = tr(M_c)
        Delta log det A     = tr(A^-1 Delta A) - sum_c tr(M_c M_c)

    For complex matrices these are the derivatives of ``log det A``, whose real
    part is ``log|det A|`` and whose imaginary part is the phase of the sign.

    Args:
        A: Forward laplacian array of the matrix, shape (..., n, n).

    Returns:
        Tuple of sign and the FwdLaplArray of log|det A|. The sign is an array for
        real matrices and a FwdLaplArray for complex ones.
    """
    sign, logdet = slogdet(A.x)
    A_inv = jnp.linalg.inv(A.x)
    M = jnp.einsum('...ij,k...jd->k...id', A_inv, A.jacobian.dense_array)
    jacobian = jnp.einsum('k...ii->k...', M)
    laplacian = jnp.einsum('...ji,...ij->...', A_inv, A.laplacian)
    laplacian -= jnp.einsum('k...id,k...di->...', M, M)
    if is_tree_complex(A.x):
        # sign = exp(i phase) with phase = Im log det A
        phase, phase_lapl = jacobian.imag, laplacian.imag
        sign = FwdLaplArray(
            sign,
            FwdJacobian.from_dense(1.0j * sign * phase),
            sign * (1.0j * phase_lapl - jnp.einsum('k...,k...->...', phase, phase)),
        )
        jacobian, laplacian = jacobian.real, laplacian.real
    return sign, FwdLaplArray(logdet, FwdJacobian.from_dense(jacobian), laplacian)


def slogdet_wrapper(
    x: tuple[ArrayOrFwdLaplArray],
    kwargs: dict[str, Any],
    sparsity_threshold: int,
):
    A = x[0]
    if not isinstance(A, FwdLaplArray):
        return slogdet(A)
    result = sparse_slogdet(A)
    if result is not None:
        return result
    return dense_slogdet(A)
