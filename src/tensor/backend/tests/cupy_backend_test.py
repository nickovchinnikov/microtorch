import unittest
import numpy as np
import cupy as cp

from src.tensor.backend.cupy import CuPyBackend, CuPyLinalg
from src.tensor.backend.types import DType


class TestCuPyBackend(unittest.TestCase):
    def setUp(self):
        self.backend = CuPyBackend()
        self.x = cp.array([[1.0, 2.0], [3.0, 4.0]])
        self.y = cp.array([[2.0, 2.0], [2.0, 2.0]])

    def test_array(self):
        out = self.backend.array([[1, 2]])
        self.assertTrue(isinstance(out, cp.ndarray))

    def test_fill_and_copy(self):
        f = self.backend.fill(self.x, 7.0)
        z = self.backend.zeros_like(self.x)
        o = self.backend.ones_like(self.x)
        c = self.backend.copy(self.x)
        self.assertTrue(cp.all(f == 7.0))
        self.assertTrue(cp.all(z == 0.0))
        self.assertTrue(cp.all(o == 1.0))
        self.assertTrue(cp.all(c == self.x))

    def test_type_casting(self):
        arr = self.backend.array([1, 2, 3])
        cast = self.backend.astype(arr, cp.float32)
        self.assertEqual(cast.dtype, cp.float32)

    def test_numpy_conversion(self):
        out = self.backend.to_numpy(self.x)
        self.assertTrue(isinstance(out, np.ndarray))
        back = self.backend.from_numpy(out)
        self.assertTrue(isinstance(back, cp.ndarray))
        self.assertTrue(np.allclose(cp.asnumpy(back), out))

    def test_get(self):
        result = self.backend.get(self.x)
        self.assertTrue(np.array_equal(result, cp.asnumpy(self.x)))

    def test_elementwise(self):
        a = self.backend.add(self.x, self.y)
        s = self.backend.subtract(self.x, self.y)
        m = self.backend.multiply(self.x, self.y)
        d = self.backend.true_divide(self.x, self.y)
        p = self.backend.pow(self.x, 2)
        self.assertTrue(cp.allclose(a, self.x + self.y))
        self.assertTrue(cp.allclose(s, self.x - self.y))
        self.assertTrue(cp.allclose(m, self.x * self.y))
        self.assertTrue(cp.allclose(d, self.x / self.y))
        self.assertTrue(cp.allclose(p, self.x**2))

    def test_unary(self):
        self.assertTrue(cp.allclose(self.backend.abs(-self.x), cp.abs(-self.x)))
        self.assertTrue(cp.allclose(self.backend.exp(self.x), cp.exp(self.x)))
        self.assertTrue(cp.allclose(self.backend.log(self.x), cp.log(self.x)))
        self.assertTrue(cp.allclose(self.backend.tanh(self.x), cp.tanh(self.x)))
        self.assertTrue(cp.allclose(self.backend.clip(self.x, 1.5, 3.5), cp.clip(self.x, 1.5, 3.5)))

    def test_comparisons(self):
        self.assertTrue(cp.all(self.backend.equal(self.x, self.y) == (self.x == self.y)))
        self.assertTrue(cp.all(self.backend.not_equal(self.x, self.y) == (self.x != self.y)))
        self.assertTrue(cp.all(self.backend.greater(self.x, self.y) == (self.x > self.y)))
        self.assertTrue(cp.all(self.backend.less_equal(self.x, self.y) == (self.x <= self.y)))

    def test_reductions(self):
        self.assertEqual(self.backend.sum(self.x).item(), cp.sum(self.x).item())
        self.assertEqual(self.backend.mean(self.x).item(), cp.mean(self.x).item())
        self.assertEqual(self.backend.max(self.x).item(), cp.max(self.x).item())
        self.assertEqual(self.backend.min(self.x).item(), cp.min(self.x).item())

    def test_shape_ops(self):
        self.assertEqual(self.backend.reshape(self.x, (4,)).shape, (4,))
        self.assertEqual(self.backend.flatten(self.x).shape, (4,))
        self.assertEqual(self.backend.transpose(self.x).shape, (2, 2))
        self.assertEqual(self.backend.broadcast_to(cp.array([[1], [2]]), (2, 2)).shape, (2, 2))
        self.assertEqual(self.backend.expand_dims(self.x, axis=0).shape, (1, 2, 2))
        self.assertEqual(self.backend.squeeze(cp.array([[[1]]])).shape, ())
        self.assertEqual(self.backend.view(self.x, (4,)).shape, (4,))
        self.assertEqual(self.backend.swapaxes(self.x, 0, 1).shape, (2, 2))

    def test_where(self):
        result = self.backend.where(self.x > 2, self.x, self.y)
        self.assertTrue(cp.allclose(result, cp.where(self.x > 2, self.x, self.y)))

    def test_random_ops(self):
        u = self.backend.random_uniform(0, 1, (2, 2))
        n = self.backend.random_normal(0, 1, (2, 2))
        r = self.backend.random_randn((2, 2))
        self.assertEqual(u.shape, (2, 2))
        self.assertEqual(n.shape, (2, 2))
        self.assertEqual(r.shape, (2, 2))

    def test_backend_properties(self):
        self.assertEqual(self.backend.name, "cupy")
        self.assertEqual(self.backend.ndarray, cp.ndarray)


class TestCuPyLinalg(unittest.TestCase):
    def setUp(self):
        self.linalg = CuPyLinalg()

    def test_norm(self):
        a = cp.array([[3.0, 4.0]])
        result = self.linalg.norm(a)
        self.assertAlmostEqual(cp.asnumpy(result), 5.0)

    def test_inv(self):
        a = cp.array([[1.0, 2.0], [3.0, 4.0]])
        result = self.linalg.inv(a)
        cp.testing.assert_allclose(result @ a, cp.eye(2), rtol=1e-5)

    def test_det(self):
        a = cp.array([[1.0, 2.0], [3.0, 4.0]])
        result = self.linalg.det(a)
        self.assertAlmostEqual(cp.asnumpy(result), -2.0)

    def test_svd(self):
        a = cp.random.rand(4, 3)
        u, s, vh = self.linalg.svd(a)
        self.assertEqual(u.shape[0], 4)
        self.assertEqual(vh.shape[-1], 3)

    def test_eigh(self):
        a = cp.array([[2.0, -1.0], [-1.0, 2.0]])
        vals, vecs = self.linalg.eigh(a)
        self.assertEqual(vals.shape, (2,))
        self.assertEqual(vecs.shape, (2, 2))

    def test_qr(self):
        a = cp.random.randn(4, 3)
        q, r = self.linalg.qr(a)
        cp.testing.assert_allclose(q @ r, a, rtol=1e-5)

    def test_solve(self):
        a = cp.array([[3.0, 1.0], [1.0, 2.0]])
        b = cp.array([9.0, 8.0])
        x = self.linalg.solve(a, b)
        cp.testing.assert_allclose(a @ x, b)

    def test_lstsq(self):
        a = cp.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b = cp.array([1.0, 2.0, 2.0])
        x, res, rank, s = self.linalg.lstsq(a, b)
        self.assertEqual(x.shape, (2,))
        self.assertIsInstance(rank.item(), int)

    def test_matrix_power(self):
        a = cp.array([[2, 0], [0, 2]])
        p = self.linalg.matrix_power(a, 3)
        cp.testing.assert_array_equal(p, cp.linalg.matrix_power(a, 3))


if __name__ == "__main__":
    unittest.main()
