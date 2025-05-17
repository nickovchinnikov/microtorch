from types import SimpleNamespace

from .base import base_ops
from .elementwise import elementwise_ops
from .math import math_ops
from .overload import overload_ops
from .reduce import reduce_ops

func_ops = SimpleNamespace(
    **vars(base_ops),
    **vars(elementwise_ops),
    **vars(math_ops),
    **vars(overload_ops),
    **vars(reduce_ops),
)

__all__ = [
    "func_ops",
]
