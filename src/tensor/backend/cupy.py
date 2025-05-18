from collections.abc import Sequence

import cupy as cp
import numpy as np

from .types import Axis, Backend, DType, Linalg, Scalar, Shape, Vector


class CuPyLinalg(Linalg):
    def norm(
        self,
        x: Vector,
        ord: int | float | str = None,
        axis: Axis = None,
        keepdims: bool = False
    ) -> Scalar:
        return cp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def inv(self, x: Vector) -> Vector:
        return cp.linalg.inv(x)

    def det(self, x: Vector) -> Scalar:
        return cp.linalg.det(x)

    def svd(
        self,
        x: Vector,
        full_matrices: bool = True,
        compute_uv: bool = True
    ) -> tuple[Vector, ...]:
        return cp.linalg.svd(
            x, full_matrices=full_matrices, compute_uv=compute_uv
        )

    def eigh(self, x: Vector, UPLO: str = "L") -> tuple[Vector, Vector]:
        return cp.linalg.eigh(x, UPLO=UPLO)

    def qr(self, x: Vector, mode: str = "reduced") -> tuple[Vector, Vector]:
        return cp.linalg.qr(x, mode=mode)

    def solve(self, a: Vector, b: Vector) -> Vector:
        return cp.linalg.solve(a, b)

    def lstsq(
        self, a: Vector, b: Vector, rcond: float | None = None
    ) -> tuple[Vector, Vector | list, Vector | int, Vector]:
        x, residuals, rank, s = cp.linalg.lstsq(a, b, rcond=rcond)
        return x, residuals, rank, s

    def matrix_power(self, x: Vector, n: int) -> Vector:
        return cp.linalg.matrix_power(x, n)


class CuPyBackend(Backend):
    def array(self, data: Sequence | Vector, dtype: cp.dtype | None = None) -> Vector:
        return cp.array(data, dtype=dtype if dtype else None)

    def fill(self, a: Vector, value: Scalar) -> Vector:
        return cp.full_like(a, value)

    def zeros_like(self, a: Vector, dtype: cp.dtype | None = None) -> Vector:
        return cp.zeros_like(a, dtype=dtype if dtype else None)

    def ones_like(self, a: Vector, dtype: cp.dtype | None = None) -> Vector:
        return cp.ones_like(a, dtype=dtype if dtype else None)

    def copy(self, a: Vector) -> Vector:
        return cp.copy(a)

    def astype(self, a: Vector, dtype: cp.dtype) -> Vector:
        return a.astype(dtype)

    def from_numpy(self, a: np.ndarray) -> Vector:
        return cp.asarray(a)

    def to_numpy(self, a: Vector) -> np.ndarray:
        return cp.asnumpy(a)

    def get(self, a: Vector) -> np.ndarray:
        return cp.asnumpy(a)

    def add(self, a: Vector, b: Scalar | Vector) -> Vector:
        return cp.add(a, b)

    def subtract(self, a: Vector, b: Scalar | Vector) -> Vector:
        return cp.subtract(a, b)

    def multiply(self, a: Vector, b: Scalar | Vector) -> Vector:
        return cp.multiply(a, b)

    def true_divide(self, a: Vector, b: Scalar | Vector) -> Vector:
        return cp.true_divide(a, b)

    def pow(self, a: Vector, exponent: Scalar | Vector) -> Vector:
        return cp.power(a, exponent)

    def abs(self, a: Vector) -> Vector:
        return cp.abs(a)

    def exp(self, a: Vector) -> Vector:
        return cp.exp(a)

    def log(self, a: Vector) -> Vector:
        return cp.log(a)

    def tanh(self, a: Vector) -> Vector:
        return cp.tanh(a)

    def clip(self, a: Vector, min_value: Scalar, max_value: Scalar) -> Vector:
        return cp.clip(a, min_value, max_value)

    def equal(self, a: Vector, b: Scalar | Vector) -> Vector:
        return cp.equal(a, b)

    def not_equal(self, a: Vector, b: Scalar | Vector) -> Vector:
        return cp.not_equal(a, b)

    def greater(self, a: Vector, b: Scalar | Vector) -> Vector:
        return cp.greater(a, b)

    def greater_equal(self, a: Vector, b: Scalar | Vector) -> Vector:
        return cp.greater_equal(a, b)

    def less(self, a: Vector, b: Scalar | Vector) -> Vector:
        return cp.less(a, b)

    def less_equal(self, a: Vector, b: Scalar | Vector) -> Vector:
        return cp.less_equal(a, b)

    def sum(self, a: Vector, axis: Axis = None, keepdims: bool = False) -> Vector:
        return cp.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a: Vector, axis: Axis = None, keepdims: bool = False) -> Vector:
        return cp.mean(a, axis=axis, keepdims=keepdims)

    def max(self, a: Vector, axis: Axis = None, keepdims: bool = False) -> Vector:
        return cp.max(a, axis=axis, keepdims=keepdims)

    def min(self, a: Vector, axis: Axis = None, keepdims: bool = False) -> Vector:
        return cp.min(a, axis=axis, keepdims=keepdims)

    def reshape(self, a: Vector, shape: Shape) -> Vector:
        return cp.reshape(a, shape)

    def view(self, a: Vector, shape: Shape) -> Vector:
        return cp.reshape(a, shape)

    def flatten(self, a: Vector) -> Vector:
        return cp.reshape(a, -1)

    def transpose(self, a: Vector, axes: Axis | None = None) -> Vector:
        return cp.transpose(a, axes)

    def broadcast_to(self, a: Vector, shape: Shape) -> Vector:
        return cp.broadcast_to(a, shape)

    def outer(self, a: Vector, b: Vector) -> Vector:
        return cp.outer(a, b)

    def swapaxes(self, a: Vector, axis1: int, axis2: int) -> Vector:
        return cp.swapaxes(a, axis1, axis2)

    def expand_dims(self, a: Vector, axis: int) -> Vector:
        return cp.expand_dims(a, axis)

    def squeeze(self, a: Vector, axis: Axis | None = None) -> Vector:
        return cp.squeeze(a, axis=axis)

    def where(self, condition: Vector, a: Vector, b: Vector) -> Vector:
        return cp.where(condition, a, b)

    def random_uniform(self, low: Scalar, high: Scalar, shape: Shape) -> Vector:
        return cp.random.uniform(low, high, shape)

    def random_normal(self, mean: Scalar, std: Scalar, shape: Shape) -> Vector:
        return cp.random.normal(mean, std, shape)

    def random_randn(self, shape: Shape) -> Vector:
        return cp.random.randn(*shape)

    @property
    def linalg(self) -> Linalg:
        return CuPyLinalg()

    @property
    def name(self) -> str:
        return "cupy"

    @property
    def ndarray(self) -> type:
        return cp.ndarray
