from enum import Enum
from typing import Union

from .backend import get_backend


class Device(str, Enum):
    r"""Enumeration for supported devices."""
    CPU = "cpu"
    CUDA = "cuda"


def check_cuda() -> bool:
    r"""Check if CuPy is installed and CUDA devices are available.

    Returns:
        bool: True if CuPy is installed and a CUDA device is available, False otherwise.
    """

    try:
        import cupy as cp
        return cp.cuda.is_available()
    except ImportError:
        return False


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

    try:
        resolved = Device(device.lower())
    except ValueError:
        raise ImportError(f"Invalid device '{device}'. Must be 'cpu' or 'cuda'.")

    try:
        get_backend(resolved)
    except Exception as e:
        raise ImportError(f"Backend for device '{resolved}' is not available: {e}")

    return resolved
