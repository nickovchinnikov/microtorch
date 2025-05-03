from typing import Optional, Tuple, Union

from src.tensor.device import Vector, _tensor
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps

Axis = Optional[Union[int, Tuple[int, ...]]]

class Reduce:
    @staticmethod
    def sum(tensor: TensorLike, axis: int = None, keepdims: bool = False) -> TProps:
        tlib = _tensor(tensor.device)
        output = tlib.sum(tensor.data, axis=axis, keepdims=keepdims)
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            def _bkwd(grad: Vector) -> Vector:
                full_grad = tlib.ones_like(tensor.data)

                if axis is None:
                    return full_grad * grad

                grad_expanded = (
                    grad if keepdims
                    else tlib.expand_dims(grad, axis=axis)
                )

                return full_grad * grad_expanded

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
    def mean(tensor: TensorLike, axis: int = None, keepdims: bool = False) -> TProps:
        count = tensor.data.shape[axis] if axis is not None else tensor.size
        return Reduce.sum(tensor, axis=axis, keepdims=keepdims) / count

    @staticmethod
    def bkwd_minmax(
        tensor: TensorLike,
        output: Vector,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
    ) -> Vector:
        tlib = _tensor(tensor.device)
        def _bkwd(grad: Vector) -> Vector:
            mask = (tensor.data == output)

            count = tlib.sum(mask) if axis is None \
                else tlib.sum(mask, axis=axis, keepdims=True)

            grad_expanded = grad if keepdims or axis is None \
                else tlib.expand_dims(grad, axis=axis)

            return mask * (grad_expanded / count)

        return _bkwd

    @staticmethod
    def max(tensor: TensorLike, axis: Axis = None, keepdims: bool = False) -> TProps:
        output = _tensor(tensor.device).max(tensor.data, axis=axis, keepdims=keepdims)
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            dependencies.append(
                Leaf(
                    value=tensor,
                    grad_fn=Reduce.bkwd_minmax(output, axis, keepdims)
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
    def min(tensor: TensorLike, axis: Axis = None, keepdims: bool = False) -> TProps:
        output = _tensor(tensor.device).min(tensor.data, axis=axis, keepdims=keepdims)
        dependencies: DependenciesList = []

        if tensor.requires_grad:
            dependencies.append(
                Leaf(
                    value=tensor,
                    grad_fn=Reduce.bkwd_minmax(output, axis, keepdims)
                )
            )

        return TProps(
            output,
            tensor.requires_grad,
            dependencies,
            device=tensor.device,
            dtype=tensor.dtype
        )
