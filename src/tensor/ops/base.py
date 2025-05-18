import numpy as np

from src.tensor.backend import Backend, Vector
from src.tensor.backend.types import Axis, Shape
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps

from .decorators import auto_backend


def bkwd_broadcast(tensor: TensorLike):
    def _bkwd(grad: Vector) -> Vector:
        # Handle scalar tensor case:
        # Original tensor was a scalar: sum all gradients
        if tensor.ndim == 0:
            return grad.sum()

        # Handle scalar grad case
        if grad.ndim == 0:
            return grad

        # Calculate the number of dimensions *added* to the tensor to achieve
        # the grad shape. This is where broadcasting might have "prepended"
        # dimensions.
        ndim_added = max(0, grad.ndim - tensor.ndim)

        if ndim_added > 0:
            grad = grad.sum(axis=tuple(range(ndim_added)), keepdims=False)

        # Sum over dimensions where tensor was broadcasted (size 1)
        reduce_axes = tuple(
            dim for dim in range(tensor.ndim)
            if tensor.shape[dim] == 1 and grad.shape[dim] > 1
        )

        if reduce_axes:
            grad = grad.sum(axis=reduce_axes, keepdims=True)

        # Ensure the final shape matches the tensor shape exactly
        if grad.shape != tensor.shape:
            grad = grad.reshape(tensor.shape)

        return grad

    return _bkwd

@auto_backend
def broadcast_to(
    tensor: TensorLike, shape: Shape, backend: Backend
) -> TProps:
    output = backend.broadcast_to(tensor.data, shape)
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        dependencies.append(
            Leaf(
                value=tensor,
                grad_fn=bkwd_broadcast(tensor)
            )
        )

    return TProps(
        output,
        tensor.requires_grad,
        dependencies,
        device=tensor.device,
        dtype=tensor.dtype
    )

@auto_backend
def view(
    tensor: TensorLike, shape: Shape, backend: Backend
) -> TProps:
    output = backend.view(tensor.data, shape)
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        def _bkwd(grad: Vector) -> Vector:
            return grad.reshape(tensor.shape)

        dependencies.append(
            Leaf(value=tensor, grad_fn=_bkwd)
        )

    return TProps(
        output,
        tensor.requires_grad,
        dependencies,
        device=tensor.device,
        dtype=tensor.dtype
    )

def reshape(
    tensor: TensorLike, shape: Shape
) -> TProps:
    return view(tensor, shape)

def transpose(
    tensor: TensorLike, axes: Axis | None = None
) -> TProps:
    output = tensor.data.transpose(axes)
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        def _bkwd(grad: Vector) -> Vector:
            # Compute the inverse permutation of axes for the backward function
            if axes is None:
                # Implicitly reverses transpose
                return grad.transpose()
            else:
                # Compute the inverse permutation of axes
                inv_axes = tuple(np.argsort(axes))
                # Transpose the gradient back using the inverse permutation
                return grad.transpose(inv_axes)

        dependencies.append(
            Leaf(value=tensor, grad_fn=_bkwd)
        )

    return TProps(
        output,
        tensor.requires_grad,
        dependencies,
        device=tensor.device,
        dtype=tensor.dtype
    )

def squeeze(
    tensor: TensorLike, axis: Axis | None = None
) -> TProps:
    output = tensor.data.squeeze(axis)
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        def _bkwd(grad: Vector) -> Vector:
            if axis is None:
                return grad.reshape(tensor.shape)
            return grad.expand_dims(axis=axis)

        dependencies.append(
            Leaf(value=tensor, grad_fn=_bkwd)
        )

    return TProps(
        output,
        tensor.requires_grad,
        dependencies,
        device=tensor.device,
        dtype=tensor.dtype
    )

@auto_backend
def unsqueeze(
    tensor: TensorLike, dim: int, backend: Backend
) -> TProps:
    output = backend.expand_dims(tensor.data, dim)
    dependencies: DependenciesList = []

    if tensor.requires_grad:
        def _bkwd(grad: Vector) -> Vector:
            return grad.squeeze(axis=dim)

        dependencies.append(
            Leaf(value=tensor, grad_fn=_bkwd)
        )

    return TProps(
        output,
        tensor.requires_grad,
        dependencies,
        device=tensor.device,
        dtype=tensor.dtype
    )


class BaseOps:
    bkwd_broadcast = staticmethod(bkwd_broadcast)
    broadcast_to = staticmethod(broadcast_to)
    reshape = staticmethod(reshape)
    squeeze = staticmethod(squeeze)
    transpose = staticmethod(transpose)
    unsqueeze = staticmethod(unsqueeze)
    view = staticmethod(view)

__all__ = [
    "BaseOps",
]
