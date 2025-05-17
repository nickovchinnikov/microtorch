from typing import Optional

from src.tensor.backend import Device, Vector
from src.tensor.types import Axis, DependenciesList, Leaf, TensorLike, TProps

from .base import Ops


class Reduce(Ops):
    def __init__(self, device: Device):
        super().__init__(device)

    def sum(
        self, tensor: TensorLike, axis: Optional[Axis] = None, keepdims: bool = False
    ) -> TProps:
        output = self.backend.sum(tensor.data, axis=axis, keepdims=keepdims)
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            def _bkwd(grad: Vector) -> Vector:
                full_grad = self.backend.ones_like(tensor.data)

                if axis is None:
                    return full_grad * grad

                grad_expanded = (
                    grad if keepdims
                    else self.backend.expand_dims(grad, axis=axis)
                )
                return full_grad * grad_expanded

            dependencies.append(Leaf(value=tensor, grad_fn=_bkwd))

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            tensor.device,
            tensor.dtype
        )

    def mean(
        self, tensor: TensorLike, axis: Optional[Axis] = None, keepdims: bool = False
    ) -> TProps:
        count = tensor.data.shape[axis] if axis is not None else tensor.size
        return tensor.sum(axis=axis, keepdims=keepdims) / count

    def _bkwd_minmax(
        self,
        tensor: TensorLike,
        output: Vector,
        axis: Optional[Axis],
        keepdims: bool = False
    ):
        def _bkwd(grad: Vector) -> Vector:
            mask = self.backend.equal(tensor.data, output)

            count = self.backend.sum(mask) if axis is None \
                else self.backend.sum(mask, axis=axis, keepdims=True)

            grad_expanded = (
                grad if keepdims or axis is None
                else self.backend.expand_dims(grad, axis=axis)
            )
            return mask * (grad_expanded / count)

        return _bkwd

    def max(
        self,
        tensor: TensorLike,
        axis: Optional[Axis] = None,
        keepdims: bool = False
    ) -> TProps:
        output = self.backend.max(tensor.data, axis=axis, keepdims=keepdims)
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            dependencies.append(
                Leaf(
                    value=tensor,
                    grad_fn=self._bkwd_minmax(tensor, output, axis, keepdims)
                )
            )

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            tensor.device,
            tensor.dtype
        )

    def min(
        self, tensor: TensorLike, axis: Optional[Axis] = None, keepdims: bool = False
    ) -> TProps:
        output = self.backend.min(tensor.data, axis=axis, keepdims=keepdims)
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            dependencies.append(
                Leaf(
                    value=tensor,
                    grad_fn=self._bkwd_minmax(tensor, output, axis, keepdims)
                )
            )

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            tensor.device,
            tensor.dtype
        )
