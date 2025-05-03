from typing import Tuple, Union

import numpy as np

from src.tensor.device import Vector, _tensor
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps


class BaseOps:
    @staticmethod
    def view(tensor: TensorLike, shape: Tuple[int, ...]) -> TProps:
        output: Vector = tensor._data.reshape(shape)
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

    @staticmethod
    def transpose(tensor: TensorLike, axes: Tuple[int, ...] = None) -> TProps:
        tnsr = _tensor(tensor.device)

        output = tnsr.transpose(tensor._data, axes=axes)
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            def _bkwd(grad: Vector) -> Vector:
                # Compute the inverse permutation of axes for the backward function
                if axes is None:
                    # Implicitly reverses transpose
                    return tnsr.transpose(grad)  
                else:
                    # Compute the inverse permutation of axes
                    inv_axes = tuple(np.argsort(axes))
                    # Transpose the gradient back using the inverse permutation
                    return tnsr.transpose(grad, axes=inv_axes)

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

    @staticmethod
    def squeeze(tensor: TensorLike, axis: Union[int, Tuple[int]]) -> TProps:
        tnsr = _tensor(tensor.device)

        output = tnsr.squeeze(tensor._data, axis=axis)
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            def _bkwd(grad: Vector) -> Vector:
                if axis is None:
                    return grad.reshape(tensor.shape)
                return tnsr.expand_dims(grad, axis=axis)
            
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

    @staticmethod
    def unsqueeze(tensor: TensorLike, dim: int) -> TProps:
        tnsr = _tensor(tensor.device)

        output = tnsr.expand_dims(tensor._data, axis=dim)
        dependencies: DependenciesList = []

        if tensor.requires_grad: 
            def _bkwd(grad: Vector) -> Vector:
                return tnsr.squeeze(grad, axis=dim)

            dependencies.append(
                Leaf(
                    value=tensor,
                    grad_fn=_bkwd
                )
            )

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            device=tensor.device,
            dtype=tensor.dtype
        )
