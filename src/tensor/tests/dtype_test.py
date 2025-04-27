import unittest

import numpy as np

from src.tensor.device import Device
from src.tensor.dtype import DTensor, DType, Leaf, get_dtype, to_dtype

# Attempt to import CuPy for CUDA support
try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


class TestDTensor(unittest.TestCase):
    r"""Unit tests for DTensor, DType mapping, and Leaf structures."""

    def setUp(self):
        r"""Common setup for all tests."""
        self.data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.device = Device.CPU
        self.dtype = DType.FLOAT32

    def test_dtype_mapping(self):
        r"""Test if get_dtype correctly maps DType enums to NumPy types."""
        self.assertEqual(get_dtype(Device.CPU, DType.FLOAT32), np.float32)

        with self.assertRaises(ValueError):
            get_dtype(Device.CPU, "invalid_dtype")  # Invalid DType input should raise ValueError

    def test_tensor_properties(self):
        r"""Test if DTensor correctly exposes its properties."""
        tensor = DTensor(
            _data=self.data,
            requires_grad=True,
            dependencies=[],
            device=self.device,
            dtype=self.dtype
        )

        self.assertEqual(tensor.ndim, 2)
        self.assertEqual(tensor.shape, (2, 2))
        self.assertEqual(tensor.size, 4)
        self.assertTrue(tensor.requires_grad)
        self.assertEqual(tensor.device, Device.CPU)
        self.assertEqual(tensor.dtype, DType.FLOAT32)

    def test_leaf_structure(self):
        r"""Test if Leaf correctly holds value and grad_fn."""
        leaf = Leaf(value=self.data, grad_fn=lambda x: x * 2)

        self.assertTrue(callable(leaf.grad_fn))
        np.testing.assert_array_equal(
            leaf.grad_fn(np.array([1, 2])),
            np.array([2, 4])
        )

    def test_tensor_props_method(self):
        r"""Test if props() correctly returns tensor metadata."""
        tensor = DTensor(
            _data=self.data,
            requires_grad=True,
            dependencies=[],
            device=self.device,
            dtype=self.dtype
        )

        props = tensor.props()

        # Props should return a tuple matching the tensor's core attributes
        self.assertEqual(props[0].tolist(), self.data.tolist())  # Compare underlying array
        self.assertTrue(props[1])                                # requires_grad
        self.assertEqual(props[2], [])                           # dependencies
        self.assertEqual(props[3], self.device)                  # device
        self.assertEqual(props[4], self.dtype)                   # dtype

    def test_to_dtype(self):
        # Test valid string inputs
        self.assertEqual(to_dtype("float32"), DType.FLOAT32)
        self.assertEqual(to_dtype("int64"), DType.INT64)
        self.assertEqual(to_dtype("int16"), DType.INT16)
        self.assertEqual(to_dtype("float64"), DType.FLOAT64)

        # Test valid np.dtype inputs
        np_dtype = np.dtype("float32")
        self.assertEqual(to_dtype(np_dtype), DType.FLOAT32)

        np_dtype = np.dtype("int64")
        self.assertEqual(to_dtype(np_dtype), DType.INT64)

        # Test valid cp.dtype inputs
        cp_dtype = cp.dtype("float32")
        self.assertEqual(to_dtype(cp_dtype), DType.FLOAT32)

        cp_dtype = cp.dtype("int64")
        self.assertEqual(to_dtype(cp_dtype), DType.INT64)

        # Test invalid string conversion
        with self.assertRaises(ValueError):
            to_dtype("invalid_dtype")

        # Test invalid numpy dtype that is not supported by DType
        np_dtype = np.dtype("complex64")
        with self.assertRaises(ValueError):
            to_dtype(np_dtype)

        # Test invalid cupy dtype that is not supported by DType
        cp_dtype = cp.dtype("complex64")
        with self.assertRaises(ValueError):
            to_dtype(cp_dtype)

        # Test empty string input
        with self.assertRaises(ValueError):
            to_dtype("")

        # Test type mismatch that is not a valid dtype (e.g., int, list)
        with self.assertRaises(ValueError):
            to_dtype(123)

        with self.assertRaises(ValueError):
            to_dtype([1, 2, 3])

if __name__ == "__main__":
    unittest.main()
