from enum import Enum
from typing import Union

import numpy as np

# Attempt to import CuPy for CUDA support
try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


# Scalar types can be int or float
Scalar = Union[int, float]

# Vector types are either NumPy or CuPy arrays
Vector = Union[np.ndarray, "cp.ndarray"]  # Type hint as string to avoid issues if CuPy is not installed.


class Device(str, Enum):
    r"""Enumeration for supported devices."""
    CPU = "cpu"
    CUDA = "cuda"


def check_cuda() -> bool:
    r"""Check if CuPy is installed and CUDA devices are available.

    Returns:
        bool: True if CuPy is installed and a CUDA device is available, False otherwise.
    """
    return CUDA_AVAILABLE


def _device(device: Union[Device, str]) -> Device:
    r"""Validates and returns the appropriate Device enum.

    Args:
        device (Union[Device, str]): The device to validate ("cpu" or "cuda").

    Raises:
        ImportError: If the device string is invalid or if CUDA is requested but unavailable.

    Returns:
        Device: The validated Device enum (Device.CPU or Device.CUDA).
    """
    if isinstance(device, Device):
        return device

    if isinstance(device, str):
        device = device.lower()

    if device not in (Device.CPU.value, Device.CUDA.value):
        raise ImportError("Device must be either 'cpu' or 'cuda'.")
    
    if device == Device.CUDA.value and not CUDA_AVAILABLE:
        raise ImportError("CUDA operations requested but CuPy is not installed!")

    return Device(device)


def _tensor(device: Union[Device, str] = Device.CPU):
    r"""Returns the appropriate tensor library based on the device.

    Args:
        device (Union[Device, str], optional): The selected device. Defaults to 'cpu'.

    Returns:
        module: NumPy if CPU, CuPy if CUDA.
    """
    device = _device(device)
    return cp if device == Device.CUDA else np

