import unittest
import numpy as np

from src.tensor.backend.numpy import NumpyBackend, NumpyLinalg


class TestNumpyBackend(unittest.TestCase):
    def setUp(self):
        self.backend = NumpyBackend()
        self.x = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.y = np.array([[2.0, 2.0], [2.0, 2.0]])

    def test_array(self):
        result = self.backend.array([[1, 2]])
        self.assertTrue(np.array_equal(result, np.array([[1, 2]])))

    def test_fill(self):
        filled = self.backend.fill(self.x, 5.0)
        np.testing.assert_array_equal(filled, np.full_like(self.x, 5.0))

    def test_zeros_ones_copy(self):
        z = self.backend.zeros_like(self.x)
        o = self.backend.ones_like(self.x)
        c = self.backend.copy(self.x)
        np.testing.assert_array_equal(z, np.zeros_like(self.x))
        np.testing.assert_array_equal(o, np.ones_like(self.x))
        np.testing.assert_array_equal(c, self.x)

    def test_type_conversion(self):
        arr = self.backend.array([1, 2, 3])
        converted = self.backend.astype(arr, np.float32)
        self.assertEqual(converted.dtype, np.float32)

    def test_numpy_conversion(self):
        out = self.backend.to_numpy(self.x)
        self.assertIsInstance(out, np.ndarray)
        out2 = self.backend.from_numpy(out)
        self.assertIsInstance(out2, np.ndarray)

    def test_get(self):
        out = self.backend.get(self.x)
        np.testing.assert_array_equal(out, self.x)

    def test_elementwise_ops(self):
        np.testing.assert_array_equal(self.backend.add(self.x, self.y), self.x + self.y)
        np.testing.assert_array_equal(self.backend.subtract(self.x, self.y), self.x - self.y)
        np.testing.assert_array_equal(self.backend.multiply(self.x, self.y), self.x * self.y)
        np.testing.assert_array_equal(self.backend.true_divide(self.x, self.y), self.x / self.y)
        np.testing.assert_array_equal(self.backend.pow(self.x, 2), self.x ** 2)

    def test_unary_ops(self):
        np.testing.assert_array_equal(self.backend.abs(-self.x), np.abs(-self.x))
        np.testing.assert_array_equal(self.backend.exp(self.x), np.exp(self.x))
        np.testing.assert_array_equal(self.backend.log(self.x), np.log(self.x))
        np.testing.assert_array_equal(self.backend.tanh(self.x), np.tanh(self.x))

    def test_comparisons(self):
        np.testing.assert_array_equal(self.backend.equal(self.x, self.y), self.x == self.y)
        np.testing.assert_array_equal(self.backend.not_equal(self.x, self.y), self.x != self.y)
        np.testing.assert_array_equal(self.backend.greater(self.x, self.y), self.x > self.y)
        np.testing.assert_array_equal(self.backend.greater_equal(self.x, self.y), self.x >= self.y)
        np.testing.assert_array_equal(self.backend.less(self.x, self.y), self.x < self.y)
        np.testing.assert_array_equal(self.backend.less_equal(self.x, self.y), self.x <= self.y)

    def test_aggregations(self):
        self.assertEqual(self.backend.sum(self.x), np.sum(self.x))
        self.assertEqual(self.backend.mean(self.x), np.mean(self.x))
        self.assertEqual(self.backend.max(self.x), np.max(self.x))
        self.assertEqual(self.backend.min(self.x), np.min(self.x))

    def test_shape_ops(self):
        reshaped = self.backend.reshape(self.x, (4,))
        np.testing.assert_array_equal(reshaped, self.x.reshape(4,))
        transposed = self.backend.transpose(self.x)
        np.testing.assert_array_equal(transposed, self.x.T)
        expanded = self.backend.expand_dims(self.x, axis=0)
        np.testing.assert_array_equal(expanded, np.expand_dims(self.x, 0))
        squeezed = self.backend.squeeze(expanded)
        np.testing.assert_array_equal(squeezed, self.x)
        flattened = self.backend.flatten(self.x)
        np.testing.assert_array_equal(flattened, self.x.flatten())
        viewed = self.backend.view(self.x, (4,))
        np.testing.assert_array_equal(viewed, self.x.reshape(4,))

    def test_misc_ops(self):
        bcast = self.backend.broadcast_to(np.array([[1], [2]]), (2, 2))
        self.assertEqual(bcast.shape, (2, 2))
        out = self.backend.outer(np.array([1, 2]), np.array([3, 4]))
        np.testing.assert_array_equal(out, np.outer([1, 2], [3, 4]))
        swapped = self.backend.swapaxes(self.x, 0, 1)
        np.testing.assert_array_equal(swapped, np.swapaxes(self.x, 0, 1))
        where_out = self.backend.where(self.x > 2, self.x, self.y)
        np.testing.assert_array_equal(where_out, np.where(self.x > 2, self.x, self.y))

    def test_random_ops(self):
        r1 = self.backend.random_uniform(0, 1, (2, 2))
        r2 = self.backend.random_normal(0, 1, (2, 2))
        r3 = self.backend.random_randn((2, 2))
        self.assertEqual(r1.shape, (2, 2))
        self.assertEqual(r2.shape, (2, 2))
        self.assertEqual(r3.shape, (2, 2))

    def test_backend_properties(self):
        self.assertEqual(self.backend.name, "numpy")
        self.assertEqual(self.backend.ndarray, np.ndarray)


class TestNumpyLinalg(unittest.TestCase):
    def setUp(self):
        self.linalg = NumpyLinalg()

    def test_norm(self):
        x = np.array([[3.0, 4.0]])
        self.assertEqual(self.linalg.norm(x), 5.0)

    def test_inv(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        inv = self.linalg.inv(a)
        np.testing.assert_array_almost_equal(inv @ a, np.eye(2))

    def test_det(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertAlmostEqual(self.linalg.det(a), np.linalg.det(a))

    def test_svd(self):
        a = np.random.randn(4, 3)
        u, s, vt = self.linalg.svd(a)
        self.assertEqual(u.shape[0], 4)
        self.assertEqual(vt.shape[-1], 3)

    def test_eigh(self):
        a = np.array([[2.0, -1.0], [-1.0, 2.0]])
        vals, vecs = self.linalg.eigh(a)
        self.assertEqual(vals.shape, (2,))
        self.assertEqual(vecs.shape, (2, 2))

    def test_qr(self):
        a = np.random.randn(4, 3)
        q, r = self.linalg.qr(a)
        np.testing.assert_array_almost_equal(q @ r, a, decimal=6)

    def test_solve(self):
        a = np.array([[3.0, 1.0], [1.0, 2.0]])
        b = np.array([9.0, 8.0])
        x = self.linalg.solve(a, b)
        np.testing.assert_array_almost_equal(a @ x, b)

    def test_lstsq(self):
        a = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0, 2.0])
        x, res, rank, s = self.linalg.lstsq(a, b)
        self.assertEqual(rank, 2)
        self.assertEqual(x.shape, (2,))

    def test_matrix_power(self):
        a = np.array([[2, 0], [0, 2]])
        powered = self.linalg.matrix_power(a, 3)
        np.testing.assert_array_equal(powered, np.linalg.matrix_power(a, 3))
