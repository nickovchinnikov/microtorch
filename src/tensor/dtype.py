from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, List, Tuple, TypeVar

from .device import Device, Vector, _tensor


class DType(Enum):
    r"""Enum representing supported data types for tensor values."""
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"


TensorType = TypeVar("Tensor")  # Generic tensor type for typing flexibility.


@dataclass(frozen=True)
class Leaf(Generic[TensorType]):
    r"""A leaf node in the computational graph.

    Attributes:
        value (TensorType): The tensor value at the leaf.
        grad_fn (Callable[[Vector], Vector]): Function to compute the gradient for this node.
    """

    value: TensorType
    grad_fn: Callable[[Vector], Vector]


DependenciesList = List[Leaf[TensorType]]  # Alias for list of dependency nodes.


@dataclass
class DProps(Generic[TensorType]):
    r"""Base properties shared by tensor classes.

    Attributes:
        _data (Vector): Raw tensor data.
        requires_grad (bool): Whether the tensor requires gradient tracking.
        dependencies (List[TensorType]): List of dependency nodes for autograd.
        device (Device): The device (CPU, GPU, etc.) where the tensor is located.
        dtype (DType): The data type of the tensor values.
    """

    _data: Vector
    requires_grad: bool
    dependencies: List[TensorType]
    device: Device
    dtype: DType


@dataclass
class DTensor(DProps[TensorType], Generic[TensorType]):
    r"""Generic tensor class that holds data and metadata.

    Inherits from:
        DProps: Base tensor properties.
    """

    # Class attributes to quickly reference dtype enums
    float64 = DType.FLOAT64
    float32 = DType.FLOAT32
    int64 = DType.INT64
    int32 = DType.INT32
    int16 = DType.INT16
    int8 = DType.INT8

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return self._data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the tensor as a tuple."""
        return self._data.shape

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return self._data.size

    @property
    def data(self) -> Vector:
        """Returns the raw underlying data of the tensor."""
        return self._data

    def props(self) -> Tuple:
        r"""Returns a tuple of properties needed to reconstruct the tensor.

        Useful for lightweight serialization or copying tensor metadata.
        
        Returns:
            Tuple: (_data, requires_grad, dependencies, device, dtype)
        """

        return (
            self._data,
            self.requires_grad,
            self.dependencies,
            self.device,
            self.dtype
        )


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

    lib = _tensor(device)  # Select the appropriate backend tensor library.

    mapping = {
        DType.FLOAT64: lib.float64,
        DType.FLOAT32: lib.float32,
        DType.INT64: lib.int64,
        DType.INT32: lib.int32,
        DType.INT16: lib.int16,
        DType.INT8: lib.int8,
    }

    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype}' for device '{device}'")

    return mapping[dtype]

