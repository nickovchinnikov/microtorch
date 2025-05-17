from src.tensor.backend import Device
from src.tensor.backend.types import Vector
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps

from .base import Ops


class Elementwise(Ops):
    def __init__(self, device: Device):
        super().__init__(device)

    def where(self, condition: TensorLike, a: TensorLike, b: TensorLike) -> TProps:
        output = self.backend.where(condition.data, a.data, b.data)
        requires_grad = a.requires_grad or b.requires_grad
        dependencies: DependenciesList = []

        if a.requires_grad:
            def _bkwd_a(grad: Vector) -> Vector:
                return self.backend.where(condition.data, grad, 0.0)
            dependencies.append(Leaf(value=a, grad_fn=_bkwd_a))

        if b.requires_grad:
            def _bkwd_b(grad: Vector) -> Vector:
                return self.backend.where(condition.data, 0.0, grad)
            dependencies.append(Leaf(value=b, grad_fn=_bkwd_b))

        return TProps(
            _data=output,
            requires_grad=requires_grad,
            dependencies=dependencies,
            device=a.device,  # assume aligned
            dtype=a.dtype,
        )

    def maximum(self, a: TensorLike, b: TensorLike) -> TProps:
        return self.where(a > b, a, b)

    def minimum(self, a: TensorLike, b: TensorLike) -> TProps:
        return self.where(a < b, a, b)

    def abs(self, tensor: TensorLike) -> TProps:
        return self.where(tensor >= 0, tensor, -tensor)
