import unittest

import numpy as np

from src.tensor.ops.reduce import ReduceOps
from src.tensor.tensor import Tensor


class TestReduceOps(unittest.TestCase):
    def test_sum_no_axis(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
        result = ReduceOps.sum(x)
        self.assertEqual(result._data, 10.0)

    def test_sum_axis_keepdims(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        result = ReduceOps.sum(x, axis=1, keepdims=True)
        np.testing.assert_allclose(result._data, [[3.0], [7.0]])
        grad = np.array([[1.0], [1.0]])
        backward = result.dependencies[0].grad_fn(grad)
        np.testing.assert_allclose(backward, np.ones_like(x.data))

    def test_sum_axis_nokeepdims(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        result = ReduceOps.sum(x, axis=1, keepdims=False)
        grad = np.array([1.0, 1.0])
        backward = result.dependencies[0].grad_fn(grad)
        np.testing.assert_allclose(backward, np.ones_like(x.data))

    def test_mean_scalar(self):
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        result = ReduceOps.mean(x)
        self.assertAlmostEqual(result.data, 2.0)
        backward = result.dependencies[0].grad_fn(np.array(1.0))
        np.testing.assert_allclose(backward, np.array([1.0, 1.0, 1.0]) / 3)

    def test_max_backward(self):
        x = Tensor(np.array([[1, 3], [5, 2]]), requires_grad=True)
        result = ReduceOps.max(x, axis=0)
        self.assertTrue(x.requires_grad)
        grad = np.array([1.0, 1.0])
        backward = result.dependencies[0].grad_fn(grad)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(backward, expected)

    def test_min_backward(self):
        x = Tensor(np.array([[2, 5], [1, 3]]), requires_grad=True)
        result = ReduceOps.min(x, axis=0)
        grad = np.array([1.0, 1.0])
        backward = result.dependencies[0].grad_fn(grad)
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(backward, expected)

    def test_max_keepdims(self):
        x = Tensor(np.array([[1, 5], [4, 2]]), requires_grad=True)
        result = ReduceOps.max(x, axis=1, keepdims=True)
        grad = np.array([[1.0], [1.0]])
        backward = result.dependencies[0].grad_fn(grad)
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_allclose(backward, expected)

    def test_min_keepdims(self):
        x = Tensor(np.array([[3, 4], [1, 2]]), requires_grad=True)
        result = ReduceOps.min(x, axis=1, keepdims=True)
        grad = np.array([[1.0], [1.0]])
        backward = result.dependencies[0].grad_fn(grad)
        expected = np.array([[1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(backward, expected)


if __name__ == "__main__":
    unittest.main()
