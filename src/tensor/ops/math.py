from src.tensor.backend.types import Backend, Vector
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps

from .decorators import auto_backend


@auto_backend
def log(tensor: TensorLike, backend: Backend) -> TProps:
    output = backend.log(tensor.data)
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        def _bkwd(grad: Vector) -> Vector:
            return grad / tensor.data
        dependencies.append(Leaf(value=tensor, grad_fn=_bkwd))

    return TProps(
        output,
        tensor.requires_grad,
        dependencies,
        device=tensor.device,
        dtype=tensor.dtype
    )

@auto_backend
def exp(tensor: TensorLike, backend: Backend) -> TProps:
    output = backend.exp(tensor.data)
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        def _bkwd(grad: Vector) -> Vector:
            return grad * output
        dependencies.append(Leaf(value=tensor, grad_fn=_bkwd))

    return TProps(
        output,
        tensor.requires_grad,
        dependencies,
        device=tensor.device,
        dtype=tensor.dtype
    )

def pow(tensor: TensorLike, pow: int) -> TProps:
    output = tensor.data**pow
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        def _bkwd(grad: Vector) -> Vector:
            return grad * (pow * (tensor.data**(pow-1)))
        dependencies.append(Leaf(value=tensor, grad_fn=_bkwd))

    return TProps(
        output,
        tensor.requires_grad,
        dependencies,
        device=tensor.device,
        dtype=tensor.dtype
    )

def sqrt(tensor: TensorLike) -> TProps:
    return pow(tensor, 0.5)

@auto_backend
def tanh(tensor: TensorLike, backend: Backend) -> TProps:
    output = backend.tanh(tensor.data)
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        def _bkwd(grad: Vector) -> Vector:
            return grad * (1 - output**2)
        dependencies.append(Leaf(value=tensor, grad_fn=_bkwd))

    return TProps(
        output,
        tensor.requires_grad,
        dependencies,
        device=tensor.device,
        dtype=tensor.dtype
    )


class MathOps:
    log = staticmethod(log)
    exp = staticmethod(exp)
    pow = staticmethod(pow)
    sqrt = staticmethod(sqrt)
    tanh = staticmethod(tanh)

__all__ = [
    "MathOps",
]
