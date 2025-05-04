import unittest

import numpy as np

from src.tensor.device import Device
from src.tensor.ops import BaseOps
from src.tensor.tensor import Tensor


class TestBaseOps(unittest.TestCase):
    def setUp(self):
        self.tensor = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device=Device.CPU)

    def test_view(self):
        result = BaseOps.view(self.tensor, (4,))
        self.assertEqual(result._data.shape, (4,))
        self.assertEqual(len(result.dependencies), 1)

    def test_reshape(self):
        result = BaseOps.reshape(self.tensor, (4,))
        self.assertEqual(result._data.shape, (4,))
        self.assertEqual(len(result.dependencies), 1)

    def test_broadcast_to(self):
        t = Tensor([[1.0], [2.0]], requires_grad=True)
        result = BaseOps.broadcast_to(t, (2, 2))
        self.assertEqual(result._data.shape, (2, 2))
        self.assertEqual(len(result.dependencies), 1)

    def test_transpose_none(self):
        result = BaseOps.transpose(self.tensor)
        self.assertEqual(result._data.shape, (2, 2))

    def test_transpose_axes(self):
        t = Tensor(np.random.rand(2, 3, 4), requires_grad=True)
        result = BaseOps.transpose(t, axes=(2, 0, 1))
        self.assertEqual(result._data.shape, (4, 2, 3))

    def test_squeeze_none(self):
        t = Tensor(np.array([[[1.0], [2.0]]]), requires_grad=True)
        result = BaseOps.squeeze(t, axis=None)
        self.assertEqual(result._data.shape, (2,))

    def test_squeeze_axis(self):
        t = Tensor(np.array([[[1.0], [2.0]]]), requires_grad=True)
        result = BaseOps.squeeze(t, axis=2)
        self.assertEqual(result._data.shape, (1, 2))

    def test_unsqueeze(self):
        t = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        result = BaseOps.unsqueeze(t, dim=1)
        self.assertEqual(result._data.shape, (2, 1))

    def test_bkwd_broadcast_scalar_tensor(self):
        t = Tensor(3.0, requires_grad=True)
        grad = np.array([[1.0, 2.0], [3.0, 4.0]])
        fn = BaseOps.bkwd_broadcast(t)
        self.assertEqual(fn(grad), grad.sum())

    def test_bkwd_broadcast_scalar_grad(self):
        t = Tensor(np.array([1.0]), requires_grad=True)
        grad = np.array(2.0)
        fn = BaseOps.bkwd_broadcast(t)
        self.assertEqual(fn(grad), grad)

    def test_bkwd_broadcast_full(self):
        t = Tensor(np.ones((1, 3)), requires_grad=True)
        grad = np.ones((2, 1, 3))
        fn = BaseOps.bkwd_broadcast(t)
        out = fn(grad)
        self.assertEqual(out.shape, (1, 3))

if __name__ == "__main__":
    unittest.main()
