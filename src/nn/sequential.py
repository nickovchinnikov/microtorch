import inspect
from collections.abc import Iterator

from src.nn.module import Module, Parameter
from src.tensor.tensor import Tensor


class Sequential(Module):
    r"""
    A sequential container of modules.
    """
    def __init__(self, *modules: Module) -> None:
        r"""
        Initialize the sequential container.

        Args:
            modules (Module): The modules to be added to the sequential container.
        """
        super().__init__()
        self.modules = modules


    def parameters(self) -> Iterator[Parameter]:
        r"""
        Method returns all parameters included in the module
        """
        # Iterate over all members of the module
        for _, value in inspect.getmembers(self):
            # Case of module tuple
            if isinstance(value, tuple):
                # Iterate over all elements in tuple
                for item in value:
                    # If instance is a module yield parameters
                    if isinstance(item, Module):
                        yield from item.parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Forward pass.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        for module in self.modules:
            x = module(x)
        return x