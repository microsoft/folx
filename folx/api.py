from enum import IntFlag
from typing import Any, Callable, NamedTuple, Protocol, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, PyTree

T = TypeVar('T', bound=PyTree[Array])
R = TypeVar('R', bound=PyTree[Array])

ExtraArgs = tuple[Array, ...]
Arrays = tuple[Array, ...]

JAC_DIM = 0  # should be either 0 or -1. TODO: switching is not support.


class FwdJacobian(NamedTuple):
    """
    Represents the Jacobian of a tensor with respect to the function's initial arguments.
    The Jacobian may either be sparse or dense. So, for a function f: R^n -> R^m, the
    Jacobian is an n x m matrix. If the Jacobian is dense, we also store it as such.
    However, it might be that the Jacobian is sparse, e.g., if f(x)=x. In such a case
    the Jacobian tensor is mostly sparse along the last dimension. Instead of explicitly
    storing all of the zeros, we store the non-zero elements and to which of the n inputs
    it depends on. So, instead of storing a n xm matrix, we store a k x m matrix, where
    k is maximum number of elements any element in m depends on. Additionally we store
    the index tensor which has shape k x m and contains integer indices between 0 and n.

    Note that we always compute sparsity patterns at compile time for efficiency reasons.
    This means that x0_idx is a numpy array and not a jax array.

    A few notes:
    - If sparsity patterns are modified in jax functions, we have to disable omnistaging.
    - Materializing the dense array is expensive and should be avoided if possible.
    - As we do not explicitly keep track of m, it might be that two dense Jacobians differ
      in the last dimension. This is not a problem as we can always pad the smaller one.
    """

    data: Array  # shape (k, ...)
    x0_idx: npt.NDArray[np.int32] | None = None  # integer array of same shape as data

    @property
    def weak(self) -> bool:
        return self.x0_idx is not None

    @property
    def unique_idx(self):
        """
        Returns an array containing the indices in that the Jacobian depends on.
        """
        if self.x0_idx is not None:
            return np.unique(self.x0_idx)
        else:
            return np.arange(self.data.shape[JAC_DIM])

    def materialize_for_idx(self, idx, max_idx: int | None = None):
        """
        Materializes the Jacobian for the given indices. If max_idx is not None, the
        resulting Jacobian will have shape (..., max_idx). Otherwise, it will have
        shape (..., max(idx) + 1). The latter only works if the idx tensor is a numpy
        array.
        """
        assert self.weak
        from .utils import broadcast_except, compact_repeated_dims_except

        # If we have static indices, we can do some optimization as we can statically
        # analyze the indices on whether they are identical along certain axes.
        # If so we can reduce these dimensions to enable coalesced memory access.
        if isinstance(idx, np.ndarray):
            idx = compact_repeated_dims_except(idx, axis=JAC_DIM)[0]

        # Broadcast to ensure shape compatibility
        x, idx = broadcast_except((self.data, idx), axis=JAC_DIM)

        # If we just broadcasted idx we should reduce these dims again.
        if isinstance(idx, np.ndarray):
            idx, copied_axes = compact_repeated_dims_except(idx, axis=JAC_DIM)
        else:
            copied_axes = ()

        indexed_axes = np.setdiff1d(np.arange(idx.ndim), (*copied_axes, JAC_DIM))
        new_order = (*indexed_axes, JAC_DIM, *copied_axes)
        inv_order = tuple(np.argsort(new_order))

        x = jnp.transpose(x, new_order)
        idx = np.transpose(idx, new_order)
        idx = idx[(..., *([0] * len(copied_axes)))]  # remove copied axes

        x_shape = (*x.shape[: len(indexed_axes)], -1, *x.shape[len(indexed_axes) + 1 :])
        x = x.reshape(
            np.prod(x.shape[: len(indexed_axes)], dtype=int),
            *x.shape[len(indexed_axes) :],
        )
        idx = idx.reshape(
            np.prod(idx.shape[: len(indexed_axes)], dtype=int),
            *idx.shape[len(indexed_axes) :],
        )

        @jax.vmap
        def aggregate(x, indices):
            return jax.ops.segment_sum(x, indices, max_idx)

        result = aggregate(x, idx).reshape(x_shape)
        result = jnp.transpose(result, inv_order)
        return result

    def get_index_mask(self, outputs):
        """
        Returns the index mask for the given outputs. The index mask is an array of
        shape broadcast(outputs, x0_idx) that contains the index of the output that
        each element in the Jacobian depends on. If an element does not depend on any
        output, the index is set to -1.
        """
        assert self.weak
        from .utils import broadcast_except

        outputs, mask = broadcast_except((outputs, self.mask), axis=JAC_DIM)

        og_shape = mask.shape[1:]
        flat_mask = mask.reshape(-1, np.prod(og_shape, dtype=int)).T
        flat_outputs = outputs.reshape(-1, np.prod(og_shape, dtype=int)).T

        @jax.vmap
        def get_indices(mask, out_mask):
            matching = mask[..., None] == out_mask
            indices = jnp.argmax(matching, axis=-1)
            indices = jnp.where(jnp.any(matching, axis=-1), indices, -1)
            return indices

        if isinstance(outputs, np.ndarray):
            with jax.ensure_compile_time_eval():
                result = np.asarray(get_indices(flat_mask, flat_outputs), dtype=int).T
        else:
            result = get_indices(flat_mask, flat_outputs).T
        return result.reshape(mask.shape)

    @property
    def data_shape(self):
        return tuple(self.data.shape[i] for i in range(self.data.ndim) if i != JAC_DIM)

    def construct_jac_for(self, idx):
        """
        Constructs the Jacobian for the given indices. If the Jacobian is dense, this
        is just a simple indexing operation. If the Jacobian is sparse, we have to
        materialize it first.
        If idx is None we return the dense matrix.
        """
        if idx is None:
            return self.dense_array
        if len(idx) == 0:
            return jnp.expand_dims(
                jnp.zeros(self.data_shape, dtype=self.data.dtype), JAC_DIM
            )
        if self.x0_idx is None:
            return self.data[idx]
        return self.materialize_for_idx(self.get_index_mask(idx), len(idx))

    @property
    def dense_array(self) -> Array:
        """
        Returns the dense Jacobian. If the Jacobian is sparse, we materialize it first.
        """
        if self.x0_idx is None:
            return self.data
        ext_idx = (..., *((None,) * len(self.data_shape)))  # this is for mypy
        return self.construct_jac_for(np.arange(self.max_n + 1)[ext_idx])

    @property
    def max_n(self) -> int:
        if self.x0_idx is not None:
            return int(np.max(self.x0_idx))
        return self.data.shape[JAC_DIM]

    @property
    def as_dense(self):
        return FwdJacobian.from_dense(self.dense_array)

    @property
    def dense_or_sparse(self) -> Array:
        return self.data

    @property
    def sparse(self) -> Array:
        assert self.weak
        return self.data

    @property
    def mask(self) -> np.ndarray:
        if self.x0_idx is not None:
            return self.x0_idx
        else:
            ext_idx = (..., *((None,) * len(self.data_shape)))  # this is for mypy
            return (
                np.ones(self.data.shape, dtype=np.int32)
                * np.arange(self.data.shape[JAC_DIM], dtype=np.int32)[ext_idx]
            )

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @classmethod
    def from_dense(cls, array):
        return cls(array, None)

    def __add__(self, other):
        assert isinstance(other, FwdJacobian)
        # If any is not weak we just add dense jacobians
        if not self.weak or not other.weak:
            from .utils import add_jacobians

            return FwdJacobian(add_jacobians(self.as_dense.data, other.as_dense.data))
        # If both are weak, we can keep them sparse
        if (other.x0_idx == self.x0_idx).all():
            return FwdJacobian(
                self.data + other.data,
                np.broadcast_arrays(self.x0_idx, other.x0_idx)[0],
            )  # type: ignore
        return FwdJacobian(
            data=jnp.concatenate((self.data, other.data), axis=JAC_DIM),
            x0_idx=np.concatenate((self.x0_idx, other.x0_idx), axis=JAC_DIM),  # type: ignore
        )

    def astype(self, dtype):
        return FwdJacobian(self.data.astype(dtype), self.x0_idx)

    @property
    def real(self):
        return FwdJacobian(self.data.real, self.x0_idx)

    @property
    def imag(self):
        return FwdJacobian(self.data.imag, self.x0_idx)

    def conj(self):
        return FwdJacobian(self.data.conj(), self.x0_idx)


class FwdLaplArray(NamedTuple):
    """
    Represents a triplet of a tensor, its Jacobian and its Laplacian with respect to
    the function's initial arguments.
    """

    x: Array
    jacobian: FwdJacobian
    laplacian: Array

    @property
    def shape(self):
        return self.x.shape

    @property
    def ndim(self):
        return self.x.ndim

    @property
    def size(self):
        return self.x.size

    @property
    def dense_jacobian(self):
        return self.jacobian.dense_array

    @property
    def is_jacobian_weak(self):
        return self.jacobian.weak

    @property
    def sparse_jacobian(self):
        return self.jacobian.sparse

    @property
    def jacobian_mask(self):
        return self.jacobian.mask

    @property
    def dense(self):
        return FwdLaplArray(self.x, self.jacobian.as_dense, self.laplacian)

    def astype(self, dtype):
        if jax.dtypes.issubdtype(dtype, jnp.floating) or jax.dtypes.issubdtype(
            dtype, jnp.complexfloating
        ):
            return FwdLaplArray(
                self.x.astype(dtype),
                self.jacobian.astype(dtype),
                self.laplacian.astype(dtype),
            )
        # If we convert to integer or boolean we drop the derivatives
        return self.x.astype(dtype)

    @property
    def dtype(self):
        return self.x.dtype

    @property
    def real(self):
        return FwdLaplArray(self.x.real, self.jacobian.real, self.laplacian.real)

    @property
    def imag(self):
        return FwdLaplArray(self.x.imag, self.jacobian.imag, self.laplacian.imag)

    def conj(self):
        return FwdLaplArray(self.x.conj(), self.jacobian.conj(), self.laplacian.conj())


def IS_LPL_ARR(x):
    return isinstance(x, FwdLaplArray)


def IS_LEAF(x):
    return isinstance(x, (FwdLaplArray, Array))


FwdLaplArrays = tuple[FwdLaplArray, ...]
ArrayOrFwdLaplArray: TypeAlias = Array | FwdLaplArray


class FwdLaplArgs(NamedTuple):
    """
    Utility class that represents a tuple of tensors, their Jacobians and their
    Laplacians with respect to the function's initial arguments.
    """

    arrays: FwdLaplArrays

    @property
    def x(self) -> Arrays:
        return tuple(a.x for a in self.arrays)

    @property
    def jacobian(self) -> tuple[FwdJacobian, ...]:
        return tuple(a.jacobian for a in self.arrays)

    @property
    def dense_jacobian(self) -> Arrays:
        return tuple(a.dense_jacobian for a in self.arrays)

    @property
    def sparse_jacobian(self) -> Arrays:
        return tuple(a.sparse_jacobian for a in self.arrays)

    @property
    def jacobian_mask(self):
        return tuple(a.jacobian_mask for a in self.arrays)

    @property
    def all_jacobian_weak(self) -> bool:
        return all(a.is_jacobian_weak for a in self.arrays)

    @property
    def any_jacobian_weak(self) -> bool:
        return any(a.is_jacobian_weak for a in self.arrays)

    @property
    def dense(self):
        return FwdLaplArgs(tuple(a.dense for a in self.arrays))

    @property
    def laplacian(self) -> Arrays:
        return tuple(a.laplacian for a in self.arrays)

    @property
    def one_hot_sparse_jacobian(self):
        jacobians = self.sparse_jacobian
        return tuple(
            tuple(
                jacobians[j]
                if i == j
                else jnp.zeros(
                    (jacobians[i].shape[0], *jacobians[j].shape[1:]),
                    dtype=jacobians[j].dtype,
                )
                for j in range(len(jacobians))
            )
            for i in range(len(jacobians))
        )

    def __len__(self) -> int:
        return len(self.arrays)


Axes = Any

ArrayOrArrays: TypeAlias = Array | tuple[Array, ...] | list[Array]
ForwardFn = Callable[..., ArrayOrArrays]


class MergeFn(Protocol):
    def __call__(self, args: Arrays, extra: ExtraArgs) -> Arrays: ...


class ForwardLaplacianFns(NamedTuple):
    forward: ForwardFn
    jvp: Callable[
        [FwdLaplArgs, dict[str, Any]], tuple[ArrayOrArrays, FwdJacobian, ArrayOrArrays]
    ]
    jac_hessian_jac_trace: Callable[[FwdLaplArgs, int], ArrayOrArrays]


class JvpFn(Protocol):
    def __call__(self, primals: Arrays, tangents: Arrays) -> tuple[Array, Array]: ...


class CustomTraceJacHessianJac(Protocol):
    def __call__(
        self,
        args: FwdLaplArgs,
        extra_args: ExtraArgs,
        merge: MergeFn,
        materialize_idx: Array,
    ) -> PyTree[Array]: ...


class ForwardLaplacian(Protocol):
    def __call__(
        self,
        args: tuple[ArrayOrFwdLaplArray],
        kwargs: dict[str, Any],
        sparsity_threshold: int,
    ) -> PyTree[ArrayOrFwdLaplArray]: ...


class FunctionFlags(IntFlag):
    GENERAL = 0
    LINEAR_IN_FIRST = 1 << 0
    LINEAR_IN_ONE = 1 << 1 | LINEAR_IN_FIRST
    LINEAR = 1 << 2 | LINEAR_IN_ONE
    REDUCTION = 1 << 3
    MULTIPLICATION = 1 << 4 | LINEAR_IN_ONE
    DOT_PRODUCT = 1 << 5 | REDUCTION | MULTIPLICATION
    INDEXING = 1 << 6 | LINEAR
    SCATTER = 1 << 7
    JOIN_JVP = 1 << 8
    SPARSE_JHJ = 1 << 9
    SUMMATION = 1 << 10 | LINEAR | REDUCTION
