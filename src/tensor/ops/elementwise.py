from src.tensor.backend import Backend
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps, Vector

from .base import auto_backend


@auto_backend
def where(
    condition: TensorLike, a: TensorLike, b: TensorLike, backend: Backend = None
) -> TProps:
    output = backend.where(condition.data, a.data, b.data)
    requires_grad = a.requires_grad or b.requires_grad
    dependencies: DependenciesList = []

    if a.requires_grad:
        def _bkwd_a(grad: Vector) -> Vector:
            return backend.where(condition.data, grad, 0.0)
        dependencies.append(Leaf(value=a, grad_fn=_bkwd_a))

    if b.requires_grad:
        def _bkwd_b(grad: Vector) -> Vector:
            return backend.where(condition.data, 0.0, grad)
        dependencies.append(Leaf(value=b, grad_fn=_bkwd_b))

    return TProps(
        _data=output,
        requires_grad=requires_grad,
        dependencies=dependencies,
        device=a.device,  # assume aligned
        dtype=a.dtype,
    )

def maximum(a: TensorLike, b: TensorLike) -> TProps:
    return where(a > b, a, b)

def minimum(a: TensorLike, b: TensorLike) -> TProps:
    return where(a < b, a, b)

def abs(tensor: TensorLike) -> TProps:
    return where(tensor >= 0, tensor, -tensor)


class ElementwiseOps:
    where=staticmethod(where)
    maximum=staticmethod(maximum)
    minimum=staticmethod(minimum)
    abs=staticmethod(abs)

__all__ = [
    "ElementwiseOps",
]
