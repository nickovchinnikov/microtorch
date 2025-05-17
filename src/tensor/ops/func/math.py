from types import SimpleNamespace

from src.tensor.backend.types import Vector
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps


def log(tensor: TensorLike) -> TProps:
    output = tensor.data.log()
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        def _bkwd(grad: Vector) -> Vector:
            return grad / tensor._data
        dependencies.append(Leaf(value=tensor, grad_fn=_bkwd))

    return TProps(
        output,
        tensor.requires_grad,
        dependencies,
        device=tensor.device,
        dtype=tensor.dtype
    )

def exp(tensor: TensorLike) -> TProps:
    output = tensor.data.exp()
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

def tanh(tensor: TensorLike) -> TProps:
    output = tensor.data.tanh()
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

math_ops = SimpleNamespace(
    log=log,
    exp=exp,
    pow=pow,
    sqrt=sqrt,
    tanh=tanh,
)
