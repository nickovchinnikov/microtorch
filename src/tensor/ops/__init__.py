from .base import BaseOps
from .elementwise import ElementwiseOps
from .math import MathOps
from .overload import OverloadOps
from .reduce import ReduceOps


class Ops:
    base_ops = BaseOps
    elementwise_ops = ElementwiseOps
    math_ops = MathOps
    overload_ops = OverloadOps
    reduce_ops = ReduceOps

__all__ = [
    "Ops",
]
