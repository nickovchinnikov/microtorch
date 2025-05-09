import unittest

import numpy as np

from src.tensor.device import Device, DType
from src.tensor.tensor import Tensor


class TestTensorCoreFields(unittest.TestCase):
    def test_tensor_init_defaults(self):
        t = Tensor([1, 2, 3])
        self.assertEqual(t.device, Device.CPU)
        self.assertEqual(t.dtype, DType.FLOAT32)
        self.assertFalse(t.requires_grad)
        self.assertIsNone(t.grad)
        np.testing.assert_array_equal(t.data, np.array([1, 2, 3], dtype=np.float32))

    def test_tensor_init_requires_grad(self):
        t = Tensor([1, 2, 3], requires_grad=True)
        self.assertTrue(t.requires_grad)
        self.assertIsNotNone(t.grad)
        np.testing.assert_array_equal(t.grad, np.zeros_like(t.data))

    def test_tensor_data_setter(self):
        t = Tensor([1, 2, 3], requires_grad=True)
        t.data = [4, 5, 6]
        np.testing.assert_array_equal(t.data, np.array([4, 5, 6], dtype=np.float32))
        self.assertEqual(t.grad.shape, (3,))

    def test_property_setters_getters(self):
        t = Tensor([1, 2, 3])
        t.requires_grad = True
        self.assertTrue(t.requires_grad)
        t.grad = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(t.grad, [1.0, 1.0, 1.0])
        t.device = Device.CPU
        t.dtype = DType.FLOAT64
        t.dependencies = []
        self.assertEqual(t.device, Device.CPU)
        self.assertEqual(t.dtype, DType.FLOAT64)
        self.assertEqual(t.dependencies, [])

    def test_shape_size_ndim_properties(self):
        t = Tensor([[1, 2], [3, 4]])
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.ndim, 2)
        self.assertEqual(t.size, 4)

    def test_repr(self):
        t = Tensor([1, 2])
        rep = repr(t)
        self.assertIn("Tensor(data=", rep)
        self.assertIn("requires_grad=", rep)
        self.assertIn("dtype=", rep)

    def test_to_device_dtype_conversion(self):
        t = Tensor([1.0, 2.0], dtype=DType.FLOAT32)
        t2 = t.to(Device.CPU, dtype=DType.FLOAT64)
        self.assertIsInstance(t2, Tensor)
        self.assertEqual(t2.dtype, DType.FLOAT64)
        np.testing.assert_array_equal(t2.data, t.data.astype(np.float64))

    def test_to_returns_self_when_same(self):
        t = Tensor([1.0], device=Device.CPU, dtype=DType.FLOAT32)
        t2 = t.to(Device.CPU, dtype=DType.FLOAT32)
        self.assertIs(t, t2)

    def test_build_ndarray_with_tensor_input(self):
        t = Tensor([1.0])
        result = Tensor.build_ndarray(t, dtype=t.dtype, device=t.device)
        np.testing.assert_array_equal(result, t.data)

    def test_from_props_reconstruction(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        t2 = Tensor.from_props(t)
        np.testing.assert_array_equal(t2.data, t.data)
        self.assertEqual(t2.requires_grad, t.requires_grad)
        self.assertEqual(t2.device, t.device)
        self.assertEqual(t2.dtype, t.dtype)

    def test_zero_grad_when_grad_exists(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        t.grad.fill(5.0)
        t.zero_grad()
        np.testing.assert_array_equal(t.grad, [0.0, 0.0])

    def test_zero_grad_when_grad_none(self):
        t = Tensor([1.0, 2.0])
        t.zero_grad()
        self.assertIsNone(t.grad)

    def test_release_grad(self):
        t = Tensor([1.0], requires_grad=True)
        t.release_grad()
        self.assertIsNone(t.grad)

if __name__ == "__main__":
    unittest.main()
