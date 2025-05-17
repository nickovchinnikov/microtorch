from typing import Callable

from src.tensor.backend import Backend, Vector
from src.tensor.types import Axis, DependenciesList, Leaf, TensorLike, TProps

from .decorators import auto_backend


@auto_backend
def sum(
    tensor: TensorLike,
    axis: Axis | None = None,
    keepdims: bool = False,
    backend: Backend = None
) -> TProps:
    output = tensor.data.sum(
        axis=axis,
        keepdims=keepdims
    )
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        def _bkwd(grad: Vector) -> Vector:
            full_grad = backend.ones_like(tensor.data)

            if axis is None:
                return full_grad * grad

            grad_expanded = (
                grad if keepdims
                else grad.expand_dims(axis=axis)
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
    tensor: TensorLike,
    axis: Axis | None = None,
    keepdims: bool = False
) -> TProps:
    count = tensor.data.shape[axis] if axis is not None else tensor.size
    return sum(tensor, axis=axis, keepdims=keepdims) / count

def bkwd_minmax(
    tensor: TensorLike,
    output: Vector,
    axis: Axis | None,
    keepdims: bool = False
) -> Callable[[Vector], Vector]:
    def _bkwd(grad: Vector) -> Vector:
        mask = tensor.data == output

        count = mask.sum() if axis is None \
            else mask.sum(axis=axis, keepdims=True)

        grad_expanded = (
            grad if keepdims or axis is None
            else grad.expand_dims(axis=axis)
        )
        return mask * (grad_expanded / count)

    return _bkwd

def max(
    tensor: TensorLike,
    axis: Axis | None = None,
    keepdims: bool = False
) -> TProps:
    output = tensor.data.max(axis=axis, keepdims=keepdims)
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        dependencies.append(
            Leaf(
                value=tensor,
                grad_fn=bkwd_minmax(tensor, output, axis, keepdims)
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
    tensor: TensorLike,
    axis: Axis | None = None,
    keepdims: bool = False
) -> TProps:
    output = tensor.data.min(axis=axis, keepdims=keepdims)
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        dependencies.append(
            Leaf(
                value=tensor,
                grad_fn=bkwd_minmax(tensor, output, axis, keepdims)
            )
        )

    return TProps(
        output,
        tensor.requires_grad,
        dependencies,
        tensor.device,
        tensor.dtype
    )

class ReduceOps:
    sum=staticmethod(sum)
    mean=staticmethod(mean)
    max=staticmethod(max)
    min=staticmethod(min)

__all__ = [
    "ReduceOps",
]
