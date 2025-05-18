import unittest

import numpy as np

from src.tensor.ops.elementwise import ElementwiseOps
from src.tensor.tensor import Tensor


class TestElementwise(unittest.TestCase):
    def test_where_with_grad(self):
        a = Tensor([[1.0, 2.0]], requires_grad=True)
        b = Tensor([[3.0, 4.0]], requires_grad=True)
        cond = Tensor([[True, False]])

        out_props = ElementwiseOps.where(cond, a, b)
        out_tensor = Tensor.from_props(out_props)

        out_tensor.backward(np.array([[1.0, 1.0]]))

        np.testing.assert_array_equal(out_tensor.data, np.where(cond.data, a.data, b.data))
        np.testing.assert_array_equal(a.grad, [[1.0, 0.0]])
        np.testing.assert_array_equal(b.grad, [[0.0, 1.0]])

    def test_where_no_grad(self):
        a = Tensor([[5.0, 6.0]])
        b = Tensor([[7.0, 8.0]])
        cond = Tensor([[False, True]])

        out = ElementwiseOps.where(cond, a, b)
        np.testing.assert_array_equal(out._data, np.where(cond.data, a.data, b.data))
        self.assertFalse(out.requires_grad)
        self.assertEqual(out.dependencies, [])

    def test_maximum(self):
        a = Tensor([[1.0, 5.0]])
        b = Tensor([[2.0, 3.0]])
        result = ElementwiseOps.maximum(a, b)
        expected = np.maximum(a.data, b.data)
        np.testing.assert_array_equal(result._data, expected)

    def test_minimum(self):
        a = Tensor([[1.0, 5.0]])
        b = Tensor([[2.0, 3.0]])
        result = ElementwiseOps.minimum(a, b)
        expected = np.minimum(a.data, b.data)
        np.testing.assert_array_equal(result._data, expected)

    def test_abs_positive_negative(self):
        a = Tensor([[-2.0, 3.0]], requires_grad=True)
        out_props = ElementwiseOps.abs(a)
        out_tensor = Tensor.from_props(out_props)

        out_tensor.backward(np.array([[1.0, 1.0]]))
        expected = np.abs(a.data)
        np.testing.assert_array_equal(out_tensor.data, expected)
        np.testing.assert_array_equal(a.grad, [[-1.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
