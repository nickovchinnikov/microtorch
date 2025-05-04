import unittest

import numpy as np

from src.tensor.ops.overload import OverloadOps
from src.tensor.tensor import Tensor


class TestOverloadOps(unittest.TestCase):
    def test_get_item_forward_backward(self):
        t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
        index = (0, slice(None))
        props = OverloadOps.get_item(t, index)
        out = props._data
        self.assertTrue(np.allclose(out, np.array([1.0, 2.0])))

        grad_out = np.array([1.0, 1.0])
        back = props.dependencies[0].grad_fn(grad_out)
        self.assertTrue(np.allclose(back, np.array([[1.0, 1.0], [0.0, 0.0]])))

    def test_neg_forward_backward(self):
        t = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)
        props = OverloadOps.neg(t)
        self.assertTrue(np.allclose(props._data, np.array([-1.0, 2.0, -3.0])))

        grad = np.array([1.0, 1.0, 1.0])
        back = props.dependencies[0].grad_fn(grad)
        self.assertTrue(np.allclose(back, -grad))

    def test_add_forward_backward(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        props = OverloadOps.add(a, b)
        self.assertTrue(np.allclose(props._data, np.array([4.0, 6.0])))

        grad = np.array([1.0, 1.0])
        ga = props.dependencies[0].grad_fn(grad)
        gb = props.dependencies[1].grad_fn(grad)
        self.assertTrue(np.allclose(ga, grad))
        self.assertTrue(np.allclose(gb, grad))

    def test_mul_forward_backward(self):
        a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
        props = OverloadOps.mul(a, b)
        self.assertTrue(np.allclose(props._data, np.array([8.0, 15.0])))

        grad = np.array([1.0, 1.0])
        ga = props.dependencies[0].grad_fn(grad)
        gb = props.dependencies[1].grad_fn(grad)
        self.assertTrue(np.allclose(ga, b.data))
        self.assertTrue(np.allclose(gb, a.data))

    def test_matmul_forward_backward_matrix(self):
        a = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
        b = Tensor(np.array([[3.0], [4.0]]), requires_grad=True)
        props = OverloadOps.matmul(a, b)
        self.assertTrue(np.allclose(props._data, np.array([[11.0]])))

        grad = np.array([[1.0]])
        ga = props.dependencies[0].grad_fn(grad)
        gb = props.dependencies[1].grad_fn(grad)
        self.assertTrue(np.allclose(ga, b.data.T))
        self.assertTrue(np.allclose(gb, a.data.T))

    def test_matmul_forward_backward_scalar(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        props = OverloadOps.matmul(a, b)
        self.assertTrue(np.allclose(props._data, np.array(11.0)))

        grad = np.array(1.0)
        ga = props.dependencies[0].grad_fn(grad)
        gb = props.dependencies[1].grad_fn(grad)
        self.assertTrue(np.allclose(ga, b.data))
        self.assertTrue(np.allclose(gb, a.data))


if __name__ == "__main__":
    unittest.main()
