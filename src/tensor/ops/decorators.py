from functools import wraps

from src.tensor.backend import get_backend
from src.tensor.types import TensorLike


def auto_backend(func):
    @wraps(func)
    def wrapper(*args, backend=None, **kwargs):
        if backend is None:
            # Find first TensorLike with a `.device` attr
            for arg in args:
                if isinstance(arg, TensorLike):
                    backend = get_backend(arg.device)
                    break
            else:
                raise ValueError("No tensor with a device to resolve backend!")
        return func(*args, backend=backend, **kwargs)
    return wrapper

__all__ = [
    "auto_backend",
]
