from typing import Callable, Iterator

from .param import Parameter


class Optimizer(object):
    """
    Base class for all optimizers.
    """

    def __init__(self, parameters: Callable[[], Iterator[Parameter]]) -> None:
        self.parameters = parameters

    def step(self) -> None:
        """
        Update the parameters of the model.
        """
        raise NotImplementedError()


class SGD(Optimizer):
    """
    A class for Stochastic Gradient Descent optimizer.
    """

    def __init__(self, parameters: Callable[[], Iterator[Parameter]], lr: float = .01) -> None:
        # Call super constructor
        super(SGD, self).__init__(parameters=parameters)
        # Save learning rate
        self.lr = lr

    def step(self) -> None:
        """
        Method performs optimization step
        """
        # Loop over all parameters
        for parameter in self.parameters():
            # Perform gradient decent
            parameter -= self.lr * parameter.grad
