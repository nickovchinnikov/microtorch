from typing import Any, Iterator

from src.tensor import Tensor
from .param import Parameter


class Module:
    r"""
    Base class for all modules.
    """

    def __init__(self) -> None:
        self.train_mode = True

    def __call__(self, *args: Any) -> Tensor:
        return self.forward(*args)

    def forward(self, *input: Any) -> Tensor:
        r"""
        Forward method to be implemented in children class

        Args:
            input (Tensor or different object): Inputs

        Returns:
            Tensor: Outputs
        """
        raise NotImplementedError()

    def train(self) -> None:
        r"""
        Set the module and all submodules to training mode
        """

        self.train_mode = True

        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                param.train()

    def eval(self) -> None:
        r"""
        Set the module and all submodules to evaluation mode
        """

        self.train_mode = False

        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                param.eval()

    def parameters(self) -> Iterator[Parameter]:
        r"""
        Returns:
            Iterator[Parameter]: Iterator of parameters
        """

        for _, item in self.__dict__.items():
            if isinstance(item, Parameter):
                yield item

            if isinstance(item, Module):
                yield from item.parameters()

    def zero_grad(self) -> None:
        r"""
        Zero the gradients of all parameters
        """

        for param in self.parameters():
            param.zero_grad()

    def params_count(self) -> int:
        r"""
        Returns:
            int: Number of parameters
        """

        num_parameters = sum(p.data.size for p in self.parameters())
        return num_parameters
