from collections.abc import Sequence

import numpy as np

from .types import Axis, Backend, DType, Linalg, Scalar, Shape, Vector


class NumpyLinalg(Linalg):
    def norm(
        self,
        x: Vector,
        ord: int | float | str = None,
        axis: Axis = None,
        keepdims: bool = False
    ) -> Scalar:
        return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def inv(self, x: Vector) -> Vector:
        return np.linalg.inv(x)

    def det(self, x: Vector) -> Scalar:
        return np.linalg.det(x)

    def svd(
        self,
        x: Vector,
        full_matrices: bool = True,
        compute_uv: bool = True,
        hermitian: bool = False
    ) -> tuple[Vector, ...]:
        return np.linalg.svd(
            x,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            hermitian=hermitian
        )

    def eig(self, x: Vector) -> tuple[Vector, Vector]:
        return np.linalg.eig(x)

    def eigh(self, x: Vector, UPLO: str = "L") -> tuple[Vector, Vector]:
        return np.linalg.eigh(x, UPLO=UPLO)

    def qr(self, x: Vector, mode: str = "reduced") -> tuple[Vector, Vector]:
        return np.linalg.qr(x, mode=mode)

    def solve(self, a: Vector, b: Vector) -> Vector:
        return np.linalg.solve(a, b)

    def lstsq(
        self,
        a: Vector,
        b: Vector,
        rcond: float | None = None
    ) -> tuple[Vector, Vector, int, Vector]:
        x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=rcond)
        return x, residuals, rank, s

    def matrix_power(self, x: Vector, n: int) -> Vector:
        return np.linalg.matrix_power(x, n)


class NumpyBackend(Backend):
    def array(self, data: Sequence | Vector, dtype: DType | None = None) -> Vector:
        return np.array(data, dtype=dtype.value if dtype else None)

    def zeros_like(self, a: Vector, dtype: DType | None = None) -> Vector:
        return np.zeros_like(a, dtype=dtype.value if dtype else None)

    def ones_like(self, a: Vector, dtype: DType | None = None) -> Vector:
        return np.ones_like(a, dtype=dtype.value if dtype else None)

    def copy(self, a: Vector) -> Vector:
        return np.copy(a)

    def astype(self, a: Vector, dtype: DType) -> Vector:
        return a.astype(dtype.value)

    def from_numpy(self, a: np.ndarray) -> Vector:
        return np.asarray(a)

    def to_numpy(self, a: Vector) -> np.ndarray:
        return np.asarray(a)

    def add(self, a: Vector, b: Scalar | Vector) -> Vector:
        return np.add(a, b)

    def subtract(self, a: Vector, b: Scalar | Vector) -> Vector:
        return np.subtract(a, b)

    def multiply(self, a: Vector, b: Scalar | Vector) -> Vector:
        return np.multiply(a, b)

    def true_divide(self, a: Vector, b: Scalar | Vector) -> Vector:
        return np.true_divide(a, b)

    def pow(self, a: Vector, exponent: Scalar | Vector) -> Vector:
        return np.power(a, exponent)

    def abs(self, a: Vector) -> Vector:
        return np.abs(a)

    def exp(self, a: Vector) -> Vector:
        return np.exp(a)

    def log(self, a: Vector) -> Vector:
        return np.log(a)

    def tanh(self, a: Vector) -> Vector:
        return np.tanh(a)

    def clip(self, a: Vector, min_value: Scalar, max_value: Scalar) -> Vector:
        return np.clip(a, min_value, max_value)

    def equal(self, a: Vector, b: Scalar | Vector) -> Vector:
        return np.equal(a, b)

    def not_equal(self, a: Vector, b: Scalar | Vector) -> Vector:
        return np.not_equal(a, b)

    def greater(self, a: Vector, b: Scalar | Vector) -> Vector:
        return np.greater(a, b)

    def greater_equal(self, a: Vector, b: Scalar | Vector) -> Vector:
        return np.greater_equal(a, b)

    def less(self, a: Vector, b: Scalar | Vector) -> Vector:
        return np.less(a, b)

    def less_equal(self, a: Vector, b: Scalar | Vector) -> Vector:
        return np.less_equal(a, b)

    def sum(self, a: Vector, axis: Axis = None, keepdims: bool = False) -> Vector:
        return np.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a: Vector, axis: Axis = None, keepdims: bool = False) -> Vector:
        return np.mean(a, axis=axis, keepdims=keepdims)

    def max(self, a: Vector, axis: Axis = None, keepdims: bool = False) -> Vector:
        return np.max(a, axis=axis, keepdims=keepdims)

    def min(self, a: Vector, axis: Axis = None, keepdims: bool = False) -> Vector:
        return np.min(a, axis=axis, keepdims=keepdims)

    def reshape(self, a: Vector, shape: Shape) -> Vector:
        return np.reshape(a, shape)

    def transpose(self, a: Vector, axes: Axis | None = None) -> Vector:
        return np.transpose(a, axes)

    def broadcast_to(self, a: Vector, shape: Shape) -> Vector:
        return np.broadcast_to(a, shape)

    def expand_dims(self, a: Vector, axis: int) -> Vector:
        return np.expand_dims(a, axis)

    def squeeze(self, a: Vector, axis: Axis | None = None) -> Vector:
        return np.squeeze(a, axis=axis)

    def where(self, condition: Vector, a: Vector, b: Vector) -> Vector:
        return np.where(condition, a, b)

    def random_uniform(self, low: Scalar, high: Scalar, shape: Shape) -> Vector:
        return np.random.uniform(low, high, shape)

    def random_normal(self, mean: Scalar, std: Scalar, shape: Shape) -> Vector:
        return np.random.normal(mean, std, shape)

    @property
    def linalg(self) -> Linalg:
        return NumpyLinalg()

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def ndarray(self) -> type:
        return np.ndarray
