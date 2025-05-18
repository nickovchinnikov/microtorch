import unittest

import numpy as np

from src.tensor.backend import Device
from src.tensor.ops.base import BaseOps
from src.tensor.tensor import Tensor


class TestBaseOps(unittest.TestCase):
    def setUp(self):
        self.tensor = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True, device=Device.CPU)

    def test_view(self):
        result = BaseOps.view(self.tensor, (4,))
        self.assertEqual(result._data.shape, (4,))
        self.assertEqual(len(result.dependencies), 1)

    def test_view_backward(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out = t.view((4,))
        grad = np.ones_like(out.data)
        out.backward(grad)
        self.assertEqual(t.grad.shape, t.shape)

    def test_reshape(self):
        result = BaseOps.reshape(self.tensor, (4,))
        self.assertEqual(result._data.shape, (4,))
        self.assertEqual(len(result.dependencies), 1)

    def test_bkwd_broadcast_mismatched_shape_triggers_reshape(self):
        t = Tensor(np.ones((1, 1, 2)), requires_grad=True)
        grad = np.ones((3, 1, 2))  # broadcasted along axis 0
        fn = BaseOps.bkwd_broadcast(t)
        out = fn(grad)
        self.assertEqual(out.shape, (1, 1, 2))

    def test_squeeze_backward_axis_none_reshape(self):
        t = Tensor(np.zeros((1, 2, 1)), requires_grad=True)
        out = Tensor.from_props(BaseOps.squeeze(t, axis=None))
        grad = np.ones_like(out.data)
        out.backward(grad)
        self.assertEqual(t.grad.shape, t.shape)

    def test_bkwd_broadcast_explicit_final_reshape(self):
        t = Tensor(np.ones((1, 1, 3)), requires_grad=True)
        grad = np.ones((2, 4, 3))  # multiple broadcasted axes
        fn = BaseOps.bkwd_broadcast(t)
        out = fn(grad)
        self.assertEqual(out.shape, (1, 1, 3))

    def test_squeeze_backward_axis_none_triggers_reshape_precisely(self):
        t = Tensor(np.ones((1, 1, 2)), requires_grad=True)
        out = Tensor.from_props(BaseOps.squeeze(t, axis=None))  # removes (1, 1) → shape (2,)
        grad = np.ones_like(out.data)
        out.backward(grad)
        self.assertEqual(t.grad.shape, t.shape)

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

    def test_transpose_backward_with_axes(self):
        t = Tensor(np.random.rand(2, 3, 4), requires_grad=True)
        out = t.transpose((2, 0, 1))  # triggers transpose axes path
        grad = np.ones_like(out.data)
        out.backward(grad)
        self.assertIsNotNone(t.grad)
        self.assertEqual(t.grad.shape, t.shape)

    def test_squeeze_none(self):
        t = Tensor(np.array([[[1.0], [2.0]]]), requires_grad=True)
        result = BaseOps.squeeze(t, axis=None)
        self.assertEqual(result._data.shape, (2,))

    def test_squeeze_axis(self):
        t = Tensor(np.array([[[1.0], [2.0]]]), requires_grad=True)
        result = BaseOps.squeeze(t, axis=2)
        self.assertEqual(result._data.shape, (1, 2))

    def test_squeeze_backward_none(self):
        t = Tensor(np.array([[[1.0], [2.0]]]), requires_grad=True)
        out = t.squeeze()
        grad = np.ones_like(out.data)
        out.backward(grad)
        self.assertEqual(t.grad.shape, t.shape)

    def test_squeeze_backward_with_axis(self):
        t = Tensor(np.array([[[1.0], [2.0]]]), requires_grad=True)
        out = t.squeeze(dim=2)
        grad = np.ones_like(out.data)
        out.backward(grad)
        self.assertEqual(t.grad.shape, t.shape)

    def test_unsqueeze(self):
        t = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        result = BaseOps.unsqueeze(t, dim=1)
        self.assertEqual(result._data.shape, (2, 1))

    def test_unsqueeze_backward(self):
        t = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        out = t.unsqueeze(dim=1)
        grad = np.ones_like(out.data)
        out.backward(grad)
        self.assertEqual(t.grad.shape, t.shape)

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

    def test_bkwd_broadcast_reduce_and_reshape(self):
        # Tensor with shape (1, 3)
        t = Tensor(np.ones((1, 3)), requires_grad=True)
        # Grad with additional dims and broadcasting needed
        grad = np.ones((2, 1, 3))
        fn = BaseOps.bkwd_broadcast(t)
        out = fn(grad)
        self.assertEqual(out.shape, (1, 3))  # Final shape match

    def test_bkwd_broadcast_full(self):
        t = Tensor(np.ones((1, 3)), requires_grad=True)
        grad = np.ones((2, 1, 3))
        fn = BaseOps.bkwd_broadcast(t)
        out = fn(grad)
        self.assertEqual(out.shape, (1, 3))

    def test_bkwd_broadcast_final_reshape(self):
        # Shape that gets broadcasted along axis 0 and 2
        t = Tensor(np.ones((1, 3)), requires_grad=True)
        grad = np.ones((2, 3, 4))  # shape doesn't match, but should reduce to (1, 3)
        fn = BaseOps.bkwd_broadcast(t)
        out = fn(grad.sum(axis=2))  # simulate reduction over axis 2
        self.assertEqual(out.shape, (1, 3))  # Valid reshape after sum
