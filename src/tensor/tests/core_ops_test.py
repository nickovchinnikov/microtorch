import os
import pickle
import tempfile
import unittest

import numpy as np

from src.tensor.device import Device, DType
from src.tensor.tensor import Tensor


class TestTensorCoreOps(unittest.TestCase):
    def test_to_returns_self(self):
        t = Tensor([1.0, 2.0], dtype=DType.FLOAT32)
        self.assertIs(t.to(Device.CPU, DType.FLOAT32), t)

    def test_to_change_dtype(self):
        t = Tensor([1.0], dtype=DType.FLOAT32)
        t2 = t.to(Device.CPU, DType.FLOAT64)
        self.assertEqual(t2.dtype, DType.FLOAT64)
        self.assertNotEqual(t2.data.dtype, t.data.dtype)

    def test_zero_and_release_grad(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        t.grad.fill(42.0)
        t.zero_grad()
        np.testing.assert_allclose(t.grad, 0.0)
        t.release_grad()
        self.assertIsNone(t.grad)

    def test_backward_scalar_grad(self):
        t = Tensor(3.0, requires_grad=True)
        t.backward()
        self.assertEqual(t.grad, 1.0)

    def test_backward_errors(self):
        t = Tensor([1.0, 2.0])
        with self.assertRaises(ValueError):
            t.backward()

        t = Tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaises(ValueError):
            t.backward()

        with self.assertRaises(ValueError):
            t.backward(np.ones((3,)))  # shape mismatch

        t = Tensor([1.0, 2.0], requires_grad=True)
        wrong_device = Tensor([1.0, 2.0], device=Device.CPU)
        wrong_device.device = "cuda"  # Fake wrong device
        with self.assertRaises(ValueError):
            t.backward(wrong_device)

    def test_clip_grad(self):
        t = Tensor([2.0, -2.0], requires_grad=True)
        t.grad = np.array([10.0, -10.0])
        t.clip_grad(clip_value=5.0)
        np.testing.assert_allclose(t.grad, [5.0, -5.0])

    def test_clip_grad_norm(self):
        t = Tensor([3.0, 4.0], requires_grad=True)
        t.grad = np.array([3.0, 4.0])
        t.clip_grad_norm(max_norm=1.0)
        expected = np.array([3.0, 4.0]) * (1.0 / (5.0 + 1e-6))
        np.testing.assert_allclose(t.grad, expected)

    def test_save_and_load_tensor(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        t.grad = np.array([0.1, 0.2])
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            path = tmp.name
        try:
            t.save(path)
            loaded = Tensor.load(path)
            np.testing.assert_array_equal(t.data, loaded.data)
            self.assertEqual(loaded.requires_grad, t.requires_grad)
            self.assertEqual(loaded.dtype, t.dtype)
            self.assertEqual(loaded.device, t.device)
        finally:
            os.remove(path)

    def test_load_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            Tensor.load("nonexistent_file.pkl")

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"not pickle")
            tmp_path = tmp.name
        with self.assertRaises(ValueError):
            Tensor.load(tmp_path)
        os.remove(tmp_path)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            pickle.dump({"bad": "format"}, tmp)
            tmp_path = tmp.name
        with self.assertRaises(ValueError):
            Tensor.load(tmp_path)
        os.remove(tmp_path)

    def test_detach(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        d = t.detach()
        self.assertFalse(d.requires_grad)
        np.testing.assert_array_equal(d.data, t.data)

    def test_clone_copy(self):
        t = Tensor([1.0, 2.0], requires_grad=True)
        c1 = t.clone()
        c2 = t.copy()
        self.assertEqual(c1.requires_grad, t.requires_grad)
        np.testing.assert_array_equal(c1.data, t.data)
        np.testing.assert_array_equal(c2.data, t.data)

    def test_repr_and_randn(self):
        t = Tensor([1.0])
        self.assertIn("Tensor(data=", repr(t))
        r = Tensor.randn((2, 2), requires_grad=True)
        self.assertEqual(r.shape, (2, 2))
        self.assertTrue(r.requires_grad)


if __name__ == "__main__":
    unittest.main()
