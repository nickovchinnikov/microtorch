from src.tensor.backend import Device
from src.tensor.backend.types import Vector
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps

from .base import Ops


class MathOps(Ops):
    def __init__(self, device: Device):
        super().__init__(device)

    def log(self, tensor: TensorLike) -> TProps:
        output = self.backend.log(tensor.data)
        dependencies: DependenciesList = []

        def _bkwd(grad: Vector) -> Vector:
            return grad / tensor._data

        if tensor.requires_grad:
            dependencies.append(Leaf(value=tensor, grad_fn=_bkwd))

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            device=tensor.device,
            dtype=tensor.dtype
        )

    def exp(self, tensor: TensorLike) -> TProps:
        output = self.backend.exp(tensor.data)
        dependencies: DependenciesList = []

        def _bkwd(grad: Vector) -> Vector:
            return grad * output

        if tensor.requires_grad:
            dependencies.append(Leaf(value=tensor, grad_fn=_bkwd))

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            device=tensor.device,
            dtype=tensor.dtype
        )

    def pow(self, tensor: TensorLike, pow: int) -> TProps:
        # Perform power operation
        output = self.backend.pow(tensor.data, pow)
        dependencies: DependenciesList = []

        def _bkwd(grad: Vector) -> Vector:
            return grad * (pow * (tensor._data**(pow-1)))

        if tensor.requires_grad:
            dependencies.append(Leaf(value=tensor, grad_fn=_bkwd))

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            device=tensor.device,
            dtype=tensor.dtype
        )

    def sqrt(self, tensor: TensorLike) -> TProps:
        return self.pow(tensor, 0.5)
    
    def tanh(self, tensor: TensorLike) -> TProps:
        output = self.backend.tanh(tensor.data)
        dependencies: DependenciesList = []

        def _bkwd(grad: Vector) -> Vector:
            return grad * (1 - output**2)

        if tensor.requires_grad:
            dependencies.append(Leaf(value=tensor, grad_fn=_bkwd))

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            device=tensor.device,
            dtype=tensor.dtype
        )
