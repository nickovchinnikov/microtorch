import unittest

import numpy as np

from src.nn.layer.linear import Linear
from src.tensor.tensor import Tensor


class TestLinear(unittest.TestCase):
    def setUp(self):
        self.in_features = 3
        self.out_features = 2
        self.linear = Linear(self.in_features, self.out_features)

    def test_initialization(self):
        self.assertEqual(self.linear.in_features, self.in_features)
        self.assertEqual(self.linear.out_features, self.out_features)
        self.assertIsNotNone(self.linear.weight)
        self.assertIsNotNone(self.linear.bias)

    def test_input_dimension_assertion(self):
        with self.assertRaises(AssertionError):
            self.linear(Tensor(np.random.randn(1)))  # 1D input
        with self.assertRaises(AssertionError):
            self.linear(Tensor(np.random.randn(1, 1, 1, 1)))  # 4D input

    def test_input_shape_mismatch(self):
        with self.assertRaises(ValueError):
            self.linear(Tensor(np.random.randn(2, 4)))  # Last dimension should be 3

    def test_2d_input_unsqueeze(self):
        x = Tensor(np.random.randn(self.out_features, self.in_features))
        output = self.linear(x)

        self.assertEqual(output.ndim, 2)
        self.assertEqual(output.shape, (self.out_features, self.out_features))

    def test_forward_pass(self):
        in_features, out_features = 3, 5
        linear = Linear(in_features, out_features)

        x = Tensor(np.random.randn(out_features, in_features))
        output = linear(x)

        self.assertEqual(output.shape, (out_features, out_features))
        self.assertIsInstance(output, Tensor)

    def test_gradient_computation(self):
        x = Tensor(np.random.randn(self.out_features, self.in_features), requires_grad=True)
        output = self.linear(x)

        # grad = Tensor.build_ndarray(np.ones_like(output.data))
        output.backward(np.ones_like(output.data))

        self.assertIsNotNone(self.linear.weight.grad)
        self.assertIsNotNone(self.linear.bias.grad)
        self.assertEqual(self.linear.weight.grad.shape, self.linear.weight.data.shape)
        self.assertEqual(self.linear.bias.grad.shape, self.linear.bias.data.shape)

    def test_zero_grad(self):
        x = Tensor(np.random.randn(2, 3))
        output = self.linear(x)
        output.backward(np.ones_like(output.data))

        self.linear.zero_grad()
        self.assertTrue(np.all(self.linear.weight.grad == 0))
        self.assertTrue(np.all(self.linear.bias.grad == 0))


if __name__ == "__main__":
    unittest.main()
