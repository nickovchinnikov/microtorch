from src.tensor import Tensor

from .module import Module


class Tanh(Module):
    r"""
    Tanh activation function.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input.tanh()
