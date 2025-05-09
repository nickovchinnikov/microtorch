import unittest

import numpy as np

from src.tensor.tensor import Tensor


class TestTensorOperatorOverloads(unittest.TestCase):
    def setUp(self):
        self.a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        self.b = Tensor([3.0, 2.0, 1.0])
        self.scalar = 2.0

    def test_getitem(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(t[0][1].item(), 2.0)

    def test_comparisons(self):
        self.assertTrue(np.all((self.a < self.b).data == np.array([True, False, False])))
        self.assertTrue(np.all((self.a > self.b).data == np.array([False, False, True])))
        self.assertTrue(np.all((self.a == self.b).data == np.array([False, True, False])))
        self.assertTrue(np.all((self.a <= self.b).data == np.array([True, True, False])))
        self.assertTrue(np.all((self.a >= self.b).data == np.array([False, True, True])))
        self.assertTrue(np.all((self.a != self.b).data == np.array([True, False, True])))

    def test_add(self):
        result = self.a + self.b
        expected = np.array([4.0, 4.0, 4.0])
        np.testing.assert_allclose(result.data, expected)

    def test_radd(self):
        result = self.scalar + self.a
        expected = np.array([3.0, 4.0, 5.0])
        np.testing.assert_allclose(result.data, expected)

    def test_iadd(self):
        a = Tensor([1.0, 2.0, 3.0])
        a += [1.0, 1.0, 1.0]
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(a.data, expected)

    def test_neg(self):
        result = -self.a
        expected = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_allclose(result.data, expected)

    def test_sub_and_rsub(self):
        result = self.a - self.b
        np.testing.assert_allclose(result.data, np.array([-2.0, 0.0, 2.0]))

        result = self.b - self.a
        np.testing.assert_allclose(result.data, np.array([2.0, 0.0, -2.0]))

    def test_isub(self):
        a = Tensor([3.0, 3.0, 3.0])
        a -= [1.0, 1.0, 1.0]
        expected = np.array([2.0, 2.0, 2.0])
        np.testing.assert_allclose(a.data, expected)

    def test_mul_and_rmul(self):
        result = self.a * self.b
        np.testing.assert_allclose(result.data, np.array([3.0, 4.0, 3.0]))

        result = self.scalar * self.a
        np.testing.assert_allclose(result.data, np.array([2.0, 4.0, 6.0]))

    def test_imul(self):
        a = Tensor([1.0, 2.0, 3.0])
        a *= [2.0, 2.0, 2.0]
        np.testing.assert_allclose(a.data, np.array([2.0, 4.0, 6.0]))

    def test_matmul_and_rmatmul(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[1.0], [1.0]])
        result = a @ b
        np.testing.assert_allclose(result.data, np.array([[3.0], [7.0]]))

        result = b.T @ a
        np.testing.assert_allclose(result.data, np.array([[4.0, 6.0]]))

    def test_pow(self):
        result = self.a ** 2
        expected = np.array([1.0, 4.0, 9.0])
        np.testing.assert_allclose(result.data, expected)

    def test_truediv_and_rtruediv(self):
        result = self.a / self.b
        np.testing.assert_allclose(result.data, np.array([1.0/3.0, 1.0, 3.0]))

        result = self.b / self.a
        np.testing.assert_allclose(result.data, np.array([3.0, 1.0, 1.0/3.0]))

    def test_itruediv(self):
        a = Tensor([2.0, 4.0, 6.0])
        a /= [2.0, 2.0, 2.0]
        np.testing.assert_allclose(a.data, np.array([1.0, 2.0, 3.0]))

if __name__ == "__main__":
    unittest.main()
