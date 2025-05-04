import unittest
from enum import Enum

import cupy as cp
import numpy as np

from src.tensor.device import Device, DType, _device, _tensor, get_dtype


class TestDevice(unittest.TestCase):
    def test_device_enum(self):
        """Test if Device Enum values are correct."""
        self.assertEqual(Device.CPU, "cpu")
        self.assertEqual(Device.CUDA, "cuda")

    def test_device_function_valid(self):
        """Test if _device function returns correct Device enum."""
        self.assertEqual(_device("cpu"), Device.CPU)
        self.assertEqual(_device(Device.CPU), Device.CPU)
        self.assertEqual(_device("cuda"), Device.CUDA)
        self.assertEqual(_device(Device.CUDA), Device.CUDA)

    def test_device_function_invalid(self):
        """Test if _device function raises an error for invalid inputs."""
        with self.assertRaises(ImportError):
            _device("tpu")
        with self.assertRaises(ImportError):
            _device("gpu")
        with self.assertRaises(ImportError):
            _device(123)

    def test_tensor_function_default(self):
        self.assertIs(_tensor(), np)  # Default is CPU

    def test_tensor_cuda_without_cupy(self):
        from src.tensor import device as devmod

        # Simulate CuPy not available
        original_flag = devmod.CUDA_AVAILABLE
        devmod.CUDA_AVAILABLE = False

        with self.assertRaises(ImportError):
            devmod._device("cuda")

        devmod.CUDA_AVAILABLE = original_flag

    def test_get_dtype_cpu(self):
        for dtype in [Device.CPU, Device.CUDA]:
            lib = _tensor(dtype)
            for enum_dtype in DType:
                self.assertEqual(get_dtype(dtype, enum_dtype), getattr(lib, enum_dtype.value))

    def test_get_dtype_invalid(self):
        class FakeDType(Enum):
            UNKNOWN = "unknown"
        with self.assertRaises(ValueError):
            get_dtype(Device.CPU, FakeDType.UNKNOWN)  # type: ignore

    def test_tensor_function(self):
        """Test if _tensor function returns the correct module."""
        self.assertIs(_tensor("cpu"), np)
        self.assertIs(_tensor("cuda"), cp)

    def test_device_function_case_insensitive(self):
        self.assertEqual(_device("CPU"), Device.CPU)
        self.assertEqual(_device("CuDa"), Device.CUDA)

if __name__ == "__main__":
    unittest.main()
