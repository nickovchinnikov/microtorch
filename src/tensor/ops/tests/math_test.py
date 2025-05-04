import unittest

import numpy as np

from src.tensor.ops.math import MathOps
from src.tensor.tensor import Tensor


class TestMathOps(unittest.TestCase):

    def test_log_forward_backward(self):
        a = Tensor([[1.0, np.e]], requires_grad=True)
        props = MathOps.log(a)
        result = Tensor.from_props(props)

        expected = np.log(a.data)
        np.testing.assert_array_almost_equal(result.data, expected)

        result.backward(np.ones_like(result.data))
        np.testing.assert_array_almost_equal(a.grad, 1 / a.data)

    def test_log_no_grad(self):
        a = Tensor([[1.0, 2.0]], requires_grad=False)
        props = MathOps.log(a)
        result = Tensor.from_props(props)
        expected = np.log(a.data)
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(props.dependencies, [])

    def test_exp_forward_backward(self):
        a = Tensor([[0.0, 1.0]], requires_grad=True)
        props = MathOps.exp(a)
        result = Tensor.from_props(props)

        expected = np.exp(a.data)
        np.testing.assert_array_almost_equal(result.data, expected)

        result.backward(np.ones_like(result.data))
        np.testing.assert_array_almost_equal(a.grad, expected)

    def test_exp_no_grad(self):
        a = Tensor([[0.0, 1.0]], requires_grad=False)
        props = MathOps.exp(a)
        result = Tensor.from_props(props)
        expected = np.exp(a.data)
        np.testing.assert_array_almost_equal(result.data, expected)
        self.assertEqual(props.dependencies, [])

    def test_pow_forward_backward(self):
        a = Tensor([[2.0, 3.0]], requires_grad=True)
        exponent = 3
        props = MathOps.pow(a, exponent)
        result = Tensor.from_props(props)

        expected = a.data ** exponent
        np.testing.assert_array_almost_equal(result.data, expected)

        result.backward(np.ones_like(result.data))
        expected_grad = exponent * (a.data ** (exponent - 1))
        np.testing.assert_array_almost_equal(a.grad, expected_grad)

    def test_pow_no_grad(self):
        a = Tensor([[2.0, 3.0]])
        result = Tensor.from_props(MathOps.pow(a, 2))
        expected = a.data ** 2
        np.testing.assert_array_equal(result.data, expected)

    def test_tanh_forward_backward(self):
        a = Tensor([[0.0, 1.0]], requires_grad=True)
        props = MathOps.tanh(a)
        result = Tensor.from_props(props)

        expected = np.tanh(a.data)
        np.testing.assert_array_almost_equal(result.data, expected)

        result.backward(np.ones_like(result.data))
        expected_grad = 1 - expected**2
        np.testing.assert_array_almost_equal(a.grad, expected_grad)

    def test_tanh_no_grad(self):
        a = Tensor([[0.0, 1.0]])
        result = Tensor.from_props(MathOps.tanh(a))
        expected = np.tanh(a.data)
        np.testing.assert_array_equal(result.data, expected)
        self.assertEqual(result.dependencies, [])


if __name__ == "__main__":
    unittest.main()
