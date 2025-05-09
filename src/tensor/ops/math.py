from src.tensor.device import Vector, _tensor
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps


class MathOps:
    @staticmethod
    def log(tensor: TensorLike) -> TProps:
        output = _tensor(tensor.device).log(tensor._data)
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

    @staticmethod
    def exp(tensor: TensorLike) -> TProps:
        output = _tensor(tensor.device).exp(tensor._data)
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

    @staticmethod
    def pow(tensor: TensorLike, pow: int) -> TProps:
        # Perform power operation
        output = tensor._data**pow
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

    @staticmethod
    def sqrt(tensor: TensorLike) -> TProps:
        return MathOps.pow(tensor, 0.5)

    @staticmethod
    def tanh(tensor: TensorLike) -> TProps:
        output = _tensor(tensor.device).tanh(tensor._data)
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
