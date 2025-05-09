import numpy as np

from src.tensor.device import Vector, _tensor
from src.tensor.types import DependenciesList, Index, Leaf, TensorLike, TProps

from .base import BaseOps


def normalize_index(index: Index):
    if hasattr(index, "data"):  # TensorLike or wrapper
        index = index.data

    # Normalize lists to numpy arrays
    if isinstance(index, list):
        return np.array(index)

    # Normalize single-item tuple
    if isinstance(index, tuple) and len(index) == 1:
        return index[0]

    return index


class OverloadOps:
    @staticmethod
    def get_item(tensor: TensorLike, index: Index) -> TProps:
        index = normalize_index(index)
        output = tensor.data[index]
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            def _bkwd(grad: Vector):
                r"""
                Backward pass for tensor indexing.
                """
                full_grad = _tensor(tensor.device).zeros_like(tensor.data)
                full_grad[index] = grad
                return full_grad

            dependencies.append(Leaf(value=tensor, grad_fn=_bkwd))

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            device=tensor.device,
            dtype=tensor.dtype
        )

    @staticmethod
    def neg(tensor: TensorLike) -> TProps:
        output = -tensor.data
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            dependencies.append(
                Leaf(value=tensor, grad_fn=lambda grad: -grad)
            )

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            device=tensor.device,
            dtype=tensor.dtype
        )

    @staticmethod
    def add(a: TensorLike, b: TensorLike) -> TProps:
        output = a.data + b.data
        requires_grad = a.requires_grad or b.requires_grad
        dependencies: DependenciesList = []

        if a.requires_grad:
            dependencies.append(
                Leaf(value=a, grad_fn=BaseOps.bkwd_broadcast(a))
            )

        if b.requires_grad:
            dependencies.append(
                Leaf(value=b, grad_fn=BaseOps.bkwd_broadcast(b))
            )

        return TProps(
            output,
            requires_grad,
            dependencies,
            device=a.device,
            dtype=a.dtype
        )

    @staticmethod
    def mul(a: TensorLike, b: TensorLike) -> TProps:
        output = a.data * b.data
        requires_grad = a.requires_grad or b.requires_grad
        dependencies: DependenciesList = []

        def _backward(a: TensorLike, b: TensorLike):
            r"""
            Backward closure function for Mul.
            """

            def _bkwd(grad: Vector) -> Vector:
                r"""
                Backward gradient function for Mul.
                """

                # Multiply grad by tensor b data
                grad = grad * b.data

                # Reduce grad to the correct shape
                return BaseOps.bkwd_broadcast(a)(grad)

            return _bkwd

        if a.requires_grad:
            dependencies.append(Leaf(value=a, grad_fn=_backward(a, b)))

        if b.requires_grad:
            dependencies.append(Leaf(value=b, grad_fn=_backward(b, a)))

        return TProps(
            output,
            requires_grad,
            dependencies,
            device=a.device,
            dtype=a.dtype
        )

    @staticmethod
    def matmul(a: TensorLike, b: TensorLike) -> TProps:
        output = a.data @ b.data
        requires_grad = a.requires_grad or b.requires_grad
        dependencies: DependenciesList = []

        if a.requires_grad:
            def bkwd_a(grad: Vector) -> Vector:
                r"""
                Backward gradient function for MatMul with respect to a.
                """
                if b.ndim > 1:
                    return grad @ b.data.swapaxes(-1, -2)
                return _tensor(a.device).outer(grad, b.data.T).squeeze()

            dependencies.append(Leaf(value=a, grad_fn=bkwd_a))

        if b.requires_grad:
            def bkwd_b(grad: Vector) -> Vector:
                r"""
                Backward gradient function for MatMul with respect to b.
                """
                if a.ndim > 1:
                    return a.data.swapaxes(-1, -2) @ grad
                return _tensor(a.device).outer(a.data.T, grad).squeeze()

            dependencies.append(Leaf(value=b, grad_fn=bkwd_b))

        return TProps(
            output,
            requires_grad,
            dependencies,
            device=a.device,
            dtype=a.dtype
        )
