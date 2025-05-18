from .numpy import NumpyBackend
from .types import Backend, Device, DType


def device_cast(device: Device | str) -> Device:
    return Device(device) if isinstance(device, str) else device

_backends: dict[Device, Backend] = {
    Device.CPU: NumpyBackend(),
}

def check_cuda() -> bool:
    r"""Check if CuPy is installed and CUDA devices are available.

    Returns:
        bool: True if CuPy is installed and a CUDA device is available, False otherwise.
    """

    try:
        import cupy as cp
        return cp.cuda.is_available() # type: ignore
    except ImportError:
        return False


if check_cuda():
    from .cupy import CuPyBackend
    _backends[Device.CUDA] = CuPyBackend()


def get_backend(device: Device) -> Backend:
    try:
        return _backends[device]
    except KeyError as err:
        raise ValueError(
            f"Backend for device '{device}' is not registered or supported."
        ) from err


def get_dtype(device: Device, dtype: DType):
    r"""Returns the correct low-level data type function for a given device.

    Args:
        device (Device): The computational device (e.g., CPU or GPU).
        dtype (DType): The desired data type.

    Raises:
        ValueError: If the dtype is unsupported for the given device.

    Returns:
        Callable: The function or dtype handler corresponding to the device and dtype.
    """

    backend = get_backend(device)
    dummy = backend.array([0], dtype=dtype)
    return dummy.dtype
