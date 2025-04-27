import unittest

import cupy as cp
import numpy as np

from src.tensor.device import Device, _device, _tensor


class TestDeviceTensor(unittest.TestCase):
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

    def test_tensor_function(self):
        """Test if _tensor function returns the correct module."""
        self.assertIs(_tensor("cpu"), np)
        self.assertIs(_tensor("cuda"), cp)

if __name__ == "__main__":
    unittest.main()
