import numpy as np

from src.tensor.device import Vector, _tensor
from src.tensor.types import Axis, DependenciesList, Leaf, Shape, TensorLike, TProps


class BaseOps:
    @staticmethod
    def bkwd_broadcast(tensor: TensorLike):
        r"""
        Backward closure function to sum across broadcasted dimensions.
    
        When performing operations between tensors of different shapes, broadcasting is used
        to align their shapes. This function ensures that the gradients are correctly summed
        over the broadcasted dimensions during the backward pass.
        
        Args:
            tensor (TensorLike): The tensor involved in the operation, used to handle its shape
                            during backward gradient computation.
        Returns:
            _bkwd (function): A function that computes the gradient, summing over broadcasted
                            dimensions to match the original tensor's shape.
        """

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

    @staticmethod
    def broadcast_to(tensor: TensorLike, shape: Shape) -> TProps:
        tnsr = _tensor(tensor.device)
        output = tnsr.broadcast_to(tensor.data, shape)
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            dependencies.append(
                Leaf(
                    value=tensor,
                    grad_fn=BaseOps.bkwd_broadcast(tensor)
                )
            )

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            device=tensor.device,
            dtype=tensor.dtype
        )

    @staticmethod
    def view(tensor: TensorLike, shape: Shape) -> TProps:
        output: Vector = tensor.data.reshape(shape)
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
    def reshape(tensor: TensorLike, shape: Shape) -> TProps:
        return BaseOps.view(tensor, shape)

    @staticmethod
    def transpose(tensor: TensorLike, axes: Axis = None) -> TProps:
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
    def squeeze(tensor: TensorLike, axis: Axis) -> TProps:
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
