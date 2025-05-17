from types import SimpleNamespace

from src.tensor.backend import Backend
from src.tensor.backend.types import Index
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps, Vector

from .base import bkwd_broadcast
from .decorators import auto_backend


@auto_backend
def get_item(tensor: TensorLike, index: Index, backend: Backend = None) -> TProps:
    output = tensor.data[index]
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        def _bkwd(grad: Vector):
            r"""
            Backward pass for tensor indexing.
            """
            full_grad = backend.zeros_like(tensor.data)
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

def add(a: TensorLike, b: TensorLike) -> TProps:
    output = a.data + b.data
    requires_grad = a.requires_grad or b.requires_grad
    dependencies: DependenciesList = []

    if a.requires_grad:
        dependencies.append(
            Leaf(value=a, grad_fn=bkwd_broadcast(a))
        )

    if b.requires_grad:
        dependencies.append(
            Leaf(value=b, grad_fn=bkwd_broadcast(b))
        )

    return TProps(
        output,
        requires_grad,
        dependencies,
        device=a.device,
        dtype=a.dtype
    )

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
            return bkwd_broadcast(a)(grad)

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

@auto_backend
def matmul(a: TensorLike, b: TensorLike, backend: Backend = None) -> TProps:
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
            return backend.outer(grad, b.data.T).squeeze()

        dependencies.append(Leaf(value=a, grad_fn=bkwd_a))

    if b.requires_grad:
        def bkwd_b(grad: Vector) -> Vector:
            r"""
            Backward gradient function for MatMul with respect to b.
            """
            if a.ndim > 1:
                return a.data.swapaxes(-1, -2) @ grad
            return backend.outer(a.data.T, grad).squeeze()

        dependencies.append(Leaf(value=b, grad_fn=bkwd_b))

    return TProps(
        output,
        requires_grad,
        dependencies,
        device=a.device,
        dtype=a.dtype
    )

overload_ops = SimpleNamespace(
    get_item=get_item,
    neg=neg,
    add=add,
    mul=mul,
    matmul=matmul,
)
