from typing import Any, Literal, Optional, Tuple

import numpy as np

from src.tensor import Tensor
from src.tensor.device import Device

InitMethod = Literal["xavier", "he", "normal", "uniform"]


class Parameter(Tensor):
    r"""
    Foundation for models parameters.
    """

    def __init__(
        self,
        *shape: int,
        data: Optional[np.ndarray] = None,
        init_method: InitMethod = "normal",
        gain: float = 1.0,
        device: Device = Device.CPU,
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
            data = self._initialize(shape, init_method, gain, device)

        super().__init__(data=data, requires_grad=True, device=device)

    def _initialize(
        self, shape: Tuple[int, ...], method: InitMethod | Any, gain: float, device: Device
    ) -> np.ndarray:
        r"""
        Initialize the parameter data.
        """
        if method == "xavier":
            std = gain * Tensor(2.0 / sum(shape), device=device).sqrt()
            return std * Tensor.randn(shape, device=device)
        if method == "he":
            std = gain * Tensor(2.0 / shape[0], device=device).sqrt()
            return std * Tensor.randn(shape, device=device)
        if method == "normal":
            return gain * Tensor.randn(shape, device=device)
        if method == "uniform":
            return gain * Tensor.uniform(-1, 1, shape, device=device)
        raise ValueError(f"Unknown initialization method: {method}")

    def to(self, device: Device) -> "Parameter":
        t = super().to(device)
        return Parameter(
            *t.shape,
            data=t.data,
            device=device
        )
