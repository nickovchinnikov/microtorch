from typing import Any, Literal, Optional, Tuple

import numpy as np

from src.tensor import Tensor

InitMethod = Literal["xavier", "he", "normal", "uniform"]


class Parameter(Tensor):
    r"""
    Foundation for models parameters.
    """

    def __init__(
        self,
        *shape: int,
        data: Optional[np.ndarray],
        init_method: InitMethod = "normal",
        gain: float = 1.0,
    ) -> None:
        r"""
        Initialize the parameter.

        Args:
            shape (tuple of int): The shape of the parameter.
            data (np.ndarray, optional): The data of the parameter. If not \
                provided, the parameter is initialized using the initialization \
                method.
            init_method (str): The initialization method. Defaults to 'normal'. \
                Possible values are 'xavier', 'he', 'normal', 'uniform'.
            gain (float): The gain for the initialization method. Defaults to 1.0.
        """

        if data is None:
            data = self._initialize(shape, init_method, gain)

        super().__init__(data=data, requires_grad=True)

    def _initialize(
        self, shape: Tuple[int, ...], method: InitMethod | Any, gain: float
    ) -> np.ndarray:
        r"""
        Initialize the parameter data.
        """
        if method == "xavier":
            std = gain * np.sqrt(2.0 / sum(shape))
            return std * np.random.randn(*shape)
        if method == "he":
            std = gain * np.sqrt(2.0 / shape[0])
            return std * np.random.randn(*shape)
        if method == "normal":
            return gain * np.random.randn(*shape)
        if method == "uniform":
            return gain * np.random.uniform(-1, 1, size=shape)
        raise ValueError(f"Unknown initialization method: {method}")
