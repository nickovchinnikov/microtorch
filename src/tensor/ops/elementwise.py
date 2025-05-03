from src.tensor.device import Vector, _tensor
from src.tensor.types import DependenciesList, Leaf, TensorLike, TProps


class Elementwise:
    @staticmethod
    def where(condition: TensorLike, a: TensorLike, b: TensorLike) -> TProps:
        tnsr = _tensor(a.device)

        output = tnsr.where(condition.data, a.data, b.data)
        requires_grad = a.requires_grad or b.requires_grad
        dependencies: DependenciesList = []

        if a.requires_grad:
            def _bkwd_a(grad: Vector) -> Vector:
                return tnsr.where(condition.data, grad, 0.0)
            dependencies.append(Leaf(value=a, grad_fn=_bkwd_a))

        if b.requires_grad:
            def _bkwd_b(grad: Vector) -> Vector:
                return tnsr.where(condition.data, 0.0, grad)
            dependencies.append(Leaf(value=b, grad_fn=_bkwd_b))

        return TProps(
            _data=output,
            requires_grad=requires_grad,
            dependencies=dependencies,
            device=a.device,  # assume aligned
            dtype=a.dtype,
        )

    @staticmethod
    def maximum(a: TensorLike, b: TensorLike) -> TProps:
        return Elementwise.where(a > b, a, b)

    @staticmethod
    def minimum(a: TensorLike, b: TensorLike) -> TProps:
        return Elementwise.where(a < b, a, b)

    @staticmethod
    def abs(tensor: TensorLike) -> TProps:
        return Elementwise.where(tensor >= 0, tensor, -tensor)
