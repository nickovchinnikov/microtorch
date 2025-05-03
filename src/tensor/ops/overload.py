from typing import Tuple, Union

from src.tensor.device import Vector, _tensor
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps


class OverloadOps:
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
    def get_item(tensor: TensorLike, index: Union[int, slice, Tuple[Union[int, slice]], Vector]) -> TProps:
        output = tensor.data[index]
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            def _bkwd(grad: Vector):
                r"""
                Backward pass for tensor indexing.
                """
                full_grad = _tensor(grad.device).zeros_like(tensor.data)
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

    @staticmethod
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

    @staticmethod
    def add(a: TensorLike, b: TensorLike) -> TProps:
        a, b = OverloadOps.device_gate(a, b)

        output = a.data + b.data
        requires_grad = a.requires_grad or b.requires_grad
        dependencies: DependenciesList = []

        if a.requires_grad:
            dependencies.append(
                Leaf(value=a, grad_fn=OverloadOps.bkwd_broadcast(a))
            )

        if b.requires_grad:
            dependencies.append(
                Leaf(value=b, grad_fn=OverloadOps.bkwd_broadcast(b))
            )

        return TProps(
            output,
            requires_grad,
            dependencies,
            device=a.device,
            dtype=a.dtype
        )

    @staticmethod
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
                return OverloadOps.bkwd_broadcast(a)(grad)

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

    @staticmethod
    def matmul(a: TensorLike, b: TensorLike) -> TProps:
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
                return _tensor(a.device).outer(grad, b.data.T).squeeze()

            dependencies.append(Leaf(value=a, grad_fn=bkwd_a))

        if b.requires_grad:
            def bkwd_b(grad: Vector) -> Vector:
                r"""
                Backward gradient function for MatMul with respect to b.
                """
                if a.ndim > 1:
                    return a.data.swapaxes(-1, -2) @ grad
                return _tensor(a.device).outer(a.data.T, grad).squeeze()

            dependencies.append(Leaf(value=b, grad_fn=bkwd_b))

        return TProps(
            output,
            requires_grad,
            dependencies,
            device=a.device,
            dtype=a.dtype
        )
