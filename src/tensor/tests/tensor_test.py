import unittest

import numpy as np
import torch

from src.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_init(self):
        # Test initialization with numpy array
        data = np.array([1, 2, 3])
        tensor = Tensor(data)
        self.assertEqual(tensor._data.tolist(), data.tolist())
        self.assertFalse(tensor.requires_grad)

        # Test initialization with list
        data = [1, 2, 3]
        tensor = Tensor(data)
        self.assertEqual(tensor._data.tolist(), data)

        # Test initialization with scalar
        data = 5
        tensor = Tensor(data)
        self.assertEqual(tensor._data.tolist(), data)

    def test_zero_grad(self):
        # Test zero_grad method
        tensor = Tensor(np.array([1, 2, 3]), requires_grad=True)
        tensor.grad = np.array([4, 5, 6])
        tensor.zero_grad()
        self.assertEqual(tensor.grad.tolist(), [0, 0, 0])

    def test_ndim(self):
        # Test ndim property
        tensor = Tensor(np.array([1, 2, 3]))
        self.assertEqual(tensor.ndim, 1)

        tensor = Tensor(np.array([[1, 2], [3, 4]]))
        self.assertEqual(tensor.ndim, 2)

    def test_shape(self):
        # Test shape property
        tensor = Tensor(np.array([1, 2, 3]))
        self.assertEqual(tensor.shape, (3,))

        tensor = Tensor(np.array([[1, 2], [3, 4]]))
        self.assertEqual(tensor.shape, (2, 2))

    def test_build_ndarray(self):
        # Test build_ndarray method
        data = [1, 2, 3]
        tensor = Tensor(data)
        self.assertEqual(tensor._data.tolist(), data)

        data = np.array([1, 2, 3])
        tensor = Tensor(data)
        self.assertEqual(tensor._data.tolist(), data.tolist())

        data = 5
        tensor = Tensor(data)
        self.assertEqual(tensor._data.tolist(), data)

    def test_indexing(self):
        x_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
        idx_data = np.array([[1], [0], [3], [2]], dtype=np.int16)

        x = Tensor(x_data, requires_grad=True)
        idx = Tensor(idx_data)

        result = x[idx]

        # Compare with PyTorch result
        x_torch = torch.tensor(x_data, requires_grad=True)
        idx_torch = torch.tensor(idx_data, dtype=torch.int)

        result_torch = x_torch[idx_torch]

        self.assertEqual(result.data.tolist(), result_torch.tolist())

        result.backward(np.ones_like(result.data))
        result_torch.backward(torch.ones_like(result_torch))

        np.testing.assert_almost_equal(
            x.grad,
            x_torch.grad.numpy(),
            decimal=2
        )

    def test_view(self):
        x_data = np.arange(10, dtype=np.float32)
        x_shape = (2, 5)

        x = Tensor(x_data, requires_grad=True)
        result = x.view(x_shape)

        x_torch = torch.tensor(x_data, requires_grad=True)
        result_torch = x_torch.view(x_shape)

        self.assertEqual(result.data.tolist(), result_torch.tolist())

        result.backward(np.ones_like(result.data))
        result_torch.backward(torch.ones_like(result_torch))

        np.testing.assert_almost_equal(
            x.grad,
            x_torch.grad.numpy(),
            decimal=2
        )



    def test_data_property(self):
        t = Tensor([1, 2, 3])
        np.testing.assert_array_equal(t.data, np.array([1., 2., 3.]))

    def test_data_setter(self):
        t = Tensor([1, 2, 3])
        t.data = [4, 5, 6]
        np.testing.assert_array_equal(t.data, np.array([4., 5., 6.]))
        self.assertEqual(t.shape, (3,))

    def test_data_gate(self):
        # Test with Tensor
        t = Tensor([1, 2, 3])
        result = Tensor.data_gate(t)

        self.assertIsInstance(result, Tensor)
        self.assertIsInstance(t, Tensor)

        np.testing.assert_array_equal(result.data, t.data)

    def test_repr(self):
        t = Tensor([1, 2, 3], requires_grad=True)
        expected = "Tensor([1. 2. 3.], requires_grad=True, shape=(3,))"
        self.assertEqual(repr(t), expected)

    def test_transpose(self):
        # Test transpose method
        tensor = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
        result = tensor.transpose()
        self.assertEqual(result.data.tolist(), [[1, 3], [2, 4]])

        # Compare with PyTorch result
        tensor_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        result_torch = tensor_torch.transpose(0, 1)
        self.assertEqual(result.data.tolist(), result_torch.tolist())

        result.backward(np.ones_like(result.data))
        result_torch.backward(torch.ones_like(result_torch))

        np.testing.assert_almost_equal(
            tensor.grad,
            tensor_torch.grad.numpy(),
            decimal=2
        )

    def test_T(self):
        # Test T property
        tensor = Tensor(np.array([[1, 2, 3], [3, 4, 5]]))
        result = tensor.T

        # Compare with PyTorch result
        tensor_torch = torch.tensor([[1, 2, 3], [3, 4, 5]])
        result_torch = tensor_torch.T

        self.assertEqual(result._data.tolist(), result_torch.tolist())

    def test_squeeze(self):
        data = np.array([[[1, 2], [3, 4]]])

        # Test squeeze method
        tensor = Tensor(data, requires_grad=True)
        result = tensor.squeeze()

        self.assertEqual(result.data.tolist(), data.squeeze().tolist())

        # Compare with PyTorch result
        tensor_torch = torch.tensor(
            data, dtype=torch.float32, requires_grad=True
        )
        result_torch = tensor_torch.squeeze()

        self.assertEqual(result.data.tolist(), result_torch.tolist())

        result.backward(np.ones_like(result.data))
        result_torch.backward(torch.ones_like(result_torch))

        np.testing.assert_almost_equal(
            tensor.grad,
            tensor_torch.grad.numpy(),
            decimal=2
        )

    def test_unsqueeze(self):
        # Test unsqueeze method
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        result = tensor.unsqueeze(0)

        self.assertEqual(result.data.tolist(), [[[1, 2], [3, 4]]])

        # Compare with PyTorch result
        tensor_torch = torch.tensor(data, requires_grad=True)
        result_torch = tensor_torch.unsqueeze(0)

        self.assertEqual(result.data.tolist(), result_torch.tolist())

        result.backward(np.ones_like(result.data))
        result_torch.backward(torch.ones_like(result_torch))

        np.testing.assert_almost_equal(
            tensor.grad,
            tensor_torch.grad.numpy(),
            decimal=2
        )

    def test_sum(self):
        # Test sum method
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        result = tensor.sum()

        self.assertEqual(result._data.tolist(), 10)

        # Compare with PyTorch result
        tensor_torch = torch.tensor(data, requires_grad=True)
        result_torch = tensor_torch.sum()

        self.assertEqual(result._data.tolist(), result_torch.tolist())

        result.backward(np.ones_like(result.data))
        result_torch.backward(torch.ones_like(result_torch))

        np.testing.assert_almost_equal(
            tensor.grad,
            tensor_torch.grad.numpy(),
            decimal=2
        )

    def test_abs(self):
        # Test abs method
        data = np.array([[-1, 2], [3, -4]], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        result = tensor.abs()

        self.assertEqual(result._data.tolist(), [[1, 2], [3, 4]])

        # Compare with PyTorch result
        tensor_torch = torch.tensor(data, requires_grad=True)
        result_torch = tensor_torch.abs()

        self.assertEqual(result._data.tolist(), result_torch.tolist())

        result.backward(np.ones_like(result.data))
        result_torch.backward(torch.ones_like(result_torch))

        np.testing.assert_almost_equal(
            tensor.grad,
            tensor_torch.grad.numpy(),
            decimal=2
        )

    def test_log(self):
        # Test log method
        data = np.array([[-1, 2], [3, -4]], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        result = tensor.log()

        np.testing.assert_almost_equal(result.data, np.log(data))

        # Compare with PyTorch result
        tensor_torch = torch.tensor(data, requires_grad=True)
        result_torch = tensor_torch.log()
        np.testing.assert_almost_equal(result.data, result_torch.detach().numpy())

        result.backward(np.ones_like(result.data))
        result_torch.backward(torch.ones_like(result_torch))

        np.testing.assert_almost_equal(
            tensor.grad,
            tensor_torch.grad.numpy(),
            decimal=2
        )

    def test_tanh(self):
        # Test tanh method
        data = np.array([[-1, 2], [3, -4]], dtype=np.float32)
        tensor = Tensor(data, requires_grad=True)
        result = tensor.tanh()

        np.testing.assert_almost_equal(result.data, np.tanh(data))

        # Compare with PyTorch result
        tensor_torch = torch.tensor(data, requires_grad=True)
        result_torch = tensor_torch.tanh()
        np.testing.assert_almost_equal(result.data, result_torch.detach().numpy(), decimal=2)

        result.backward(np.ones_like(result.data))
        result_torch.backward(torch.ones_like(result_torch))

        np.testing.assert_almost_equal(
            tensor.grad,
            tensor_torch.grad.numpy(),
            decimal=2
        )


    def test_add(self):
        # Test add method
        tensor1 = Tensor(np.array([1, 2, 3]), requires_grad=True)
        tensor2 = Tensor(np.array([4, 5, 6]), requires_grad=True)
        result = tensor1 + tensor2
        self.assertEqual(result._data.tolist(), [5, 7, 9])

        # Compare with PyTorch result
        tensor1_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        tensor2_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        result_torch = tensor1_torch + tensor2_torch
        self.assertEqual(result._data.tolist(), result_torch.tolist())

        # Check grads for the result and compare with PyTorch result
        result.backward(np.array([1, 1, 1]))
        result_torch.backward(torch.tensor([1, 1, 1]))

        self.assertEqual(tensor1.grad.tolist(), tensor1_torch.grad.tolist())
        self.assertEqual(tensor2.grad.tolist(), tensor2_torch.grad.tolist())

    def test_radd(self):
        # Test radd method
        tensor = Tensor(np.array([1, 2, 3]), requires_grad=True)
        result = 5 + tensor
        self.assertEqual(result._data.tolist(), [6, 7, 8])

        # Compare with PyTorch result
        tensor_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result_torch = 5 + tensor_torch
        self.assertEqual(result._data.tolist(), result_torch.tolist())

        # Backward
        result.backward(np.array([1, 1, 1]))
        result_torch.backward(torch.tensor([1, 1, 1]))

        self.assertEqual(tensor.grad.tolist(), tensor_torch.grad.tolist())

    def test_iadd(self):
        # Test iadd method
        tensor = Tensor(np.array([1, 2, 3]))
        tensor += Tensor(np.array([4, 5, 6]))
        self.assertEqual(tensor._data.tolist(), [5, 7, 9])

        # Compare with PyTorch result
        tensor_torch = torch.tensor([1, 2, 3])
        tensor_torch += torch.tensor([4, 5, 6])
        self.assertEqual(tensor._data.tolist(), tensor_torch.tolist())

    def test_add_broadcast(self):
        a = Tensor([[1, 2, 3]], requires_grad=True)
        b = Tensor([[1], [2], [3]], requires_grad=True)
        c = a + b

        a_torch = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        b_torch = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        c_torch = a_torch + b_torch

        np.testing.assert_array_equal(c.data, [[2, 3, 4], [3, 4, 5], [4, 5, 6]])
        self.assertEqual(c.data.tolist(), c_torch.tolist())

        c.backward(np.ones_like(c.data))
        c_torch.backward(torch.ones_like(c_torch))

        # Gradients
        np.testing.assert_array_equal(a.grad, a_torch.grad.data.numpy())
        self.assertEqual(b.grad.data.tolist(), b_torch.grad.data.tolist())

    def test_add_scalar(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = 5
        c = [0] + a + b

        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        c_torch = torch.tensor([0]) + a_torch + b

        np.testing.assert_array_equal(c.data, [6, 7, 8])

        c.backward(np.ones_like(c.data))
        c_torch.backward(torch.ones_like(c_torch))

        np.testing.assert_array_equal(c.grad.data, [1, 1, 1])
        np.testing.assert_array_equal(a.grad.data, [1, 1, 1])

        self.assertEqual(a.grad.tolist(), a_torch.grad.tolist())

    def test_neg(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = -a

        a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        b_torch = -a_torch

        self.assertEqual(a.data.tolist(), a_torch.tolist())
        np.testing.assert_array_equal(b.data, [-1, -2, -3])

        self.assertEqual(b.data.tolist(), b_torch.tolist())
        self.assertTrue(b.requires_grad)

        b.backward(np.ones_like(a.data))
        b_torch.backward(torch.ones_like(a_torch))

        np.testing.assert_array_equal(a.grad, [-1, -1, -1])
        self.assertEqual(a.grad.tolist(), a_torch.grad.tolist())

    def test_sub(self):
        # Test sub method
        tensor1 = Tensor(np.array([1, 2, 3]), requires_grad=True)
        tensor2 = Tensor(np.array([4, 5, 6]), requires_grad=True)
        result = tensor1 - tensor2
        self.assertEqual(result._data.tolist(), [-3, -3, -3])

        # Compare with PyTorch result
        tensor1_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        tensor2_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        result_torch = tensor1_torch - tensor2_torch
        self.assertEqual(result._data.tolist(), result_torch.tolist())

        # Check grads for the result and compare with PyTorch result
        result.backward(np.array([1, 1, 1]))
        result_torch.backward(torch.tensor([1, 1, 1]))

        self.assertEqual(tensor1.grad.tolist(), tensor1_torch.grad.tolist())
        self.assertEqual(tensor2.grad.tolist(), tensor2_torch.grad.tolist())

    def test_rsub(self):
        # Test rsub method
        tensor = Tensor(np.array([1, 2, 3]), requires_grad=True)
        result = 5 - tensor
        self.assertEqual(result._data.tolist(), [4, 3, 2])

        # Compare with PyTorch result
        tensor_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result_torch = 5 - tensor_torch
        self.assertEqual(result._data.tolist(), result_torch.tolist())

        # Backward
        result.backward(np.array([1, 1, 1]))
        result_torch.backward(torch.tensor([1, 1, 1]))

        self.assertEqual(tensor.grad.tolist(), tensor_torch.grad.tolist())

    def test_isub(self):
        # Test isub method
        tensor = Tensor(np.array([1, 2, 3]))
        tensor -= Tensor(np.array([4, 5, 6]))
        self.assertEqual(tensor._data.tolist(), [-3, -3, -3])

        # Compare with PyTorch result
        tensor_torch = torch.tensor([1, 2, 3])
        tensor_torch -= torch.tensor([4, 5, 6])
        self.assertEqual(tensor._data.tolist(), tensor_torch.tolist())

    def test_mul(self):
        # Test mul method
        tensor1 = Tensor(np.array([1, 2, 3]), requires_grad=True)
        tensor2 = Tensor(np.array([4, 5, 6]), requires_grad=True)
        result = tensor1 * tensor2
        self.assertEqual(result._data.tolist(), [4, 10, 18])

        # Compare with PyTorch result
        tensor1_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        tensor2_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        result_torch = tensor1_torch * tensor2_torch
        self.assertEqual(result._data.tolist(), result_torch.tolist())

        # Check grads for the result and compare with PyTorch result
        result.backward(np.array([1, 1, 1]))
        result_torch.backward(torch.tensor([1, 1, 1]))

        self.assertEqual(tensor1.grad.tolist(), tensor1_torch.grad.tolist())
        self.assertEqual(tensor2.grad.tolist(), tensor2_torch.grad.tolist())

    def test_rmul(self):
        # Test rmul method
        tensor = Tensor(np.array([1, 2, 3]), requires_grad=True)
        result = 5 * tensor
        self.assertEqual(result._data.tolist(), [5, 10, 15])

        # Compare with PyTorch result
        tensor_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result_torch = 5 * tensor_torch
        self.assertEqual(result._data.tolist(), result_torch.tolist())

        # Backward
        result.backward(np.array([1, 1, 1]))
        result_torch.backward(torch.tensor([1, 1, 1]))

        self.assertEqual(tensor.grad.tolist(), tensor_torch.grad.tolist())

    def test_imul(self):
        # Test imul method
        tensor = Tensor(np.array([1, 2, 3]))
        tensor *= Tensor(np.array([4, 5, 6]))
        self.assertEqual(tensor._data.tolist(), [4, 10, 18])

        # Compare with PyTorch result
        tensor_torch = torch.tensor([1, 2, 3])
        tensor_torch *= torch.tensor([4, 5, 6])
        self.assertEqual(tensor._data.tolist(), tensor_torch.tolist())


    def test_matmul(self):
        # Test matmul method
        tensor1 = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
        tensor2 = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=True)
        result = tensor1 @ tensor2
        self.assertEqual(result._data.tolist(), [[19, 22], [43, 50]])

        # Compare with PyTorch result
        tensor1_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tensor2_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        result_torch = tensor1_torch @ tensor2_torch
        self.assertEqual(result._data.tolist(), result_torch.tolist())

        # Check grads for the result and compare with PyTorch result
        result.backward(np.array([[1, 1], [1, 1]]))
        result_torch.backward(torch.tensor([[1, 1], [1, 1]]))

        self.assertEqual(tensor1.grad.tolist(), tensor1_torch.grad.tolist())
        self.assertEqual(tensor2.grad.tolist(), tensor2_torch.grad.tolist())


    def test_rmatmul(self):
        # Test rmatmul method
        tensor = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
        result = tensor @ Tensor(np.array([[5, 6], [7, 8]]))
        self.assertEqual(result._data.tolist(), [[19, 22], [43, 50]])

        # Compare with PyTorch result
        tensor_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        result_torch = tensor_torch @ torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        self.assertEqual(result._data.tolist(), result_torch.tolist())

        # Backward
        result.backward(np.array([[1, 1], [1, 1]]))
        result_torch.backward(torch.tensor([[1, 1], [1, 1]]))

        self.assertEqual(tensor.grad.tolist(), tensor_torch.grad.tolist())


    def test_pow(self):
        # Test pow method
        tensor = Tensor(np.array([1, 2, 3]), requires_grad=True)
        result = tensor ** 2
        self.assertEqual(result._data.tolist(), [1, 4, 9])

        # Compare with PyTorch result
        tensor_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result_torch = tensor_torch ** 2
        self.assertEqual(result._data.tolist(), result_torch.tolist())

        # Backward
        result.backward(np.array([1, 1, 1]))
        result_torch.backward(torch.tensor([1, 1, 1]))

        self.assertEqual(tensor.grad.tolist(), tensor_torch.grad.tolist())


    def test_truediv(self):
        # Test truediv method
        tensor1 = Tensor(np.array([1, 2, 3]), requires_grad=True)
        tensor2 = Tensor(np.array([4, 5, 6]), requires_grad=True)
        result = tensor1 / tensor2

        np.testing.assert_almost_equal(
            result._data,
            np.array([0.25, 0.4, 0.5]),
            decimal=5
        )

        # Compare with PyTorch result
        tensor1_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        tensor2_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
        result_torch = tensor1_torch / tensor2_torch

        np.testing.assert_almost_equal(
            result.data, result_torch.detach().numpy(), decimal=2
        )

        # Check grads for the result and compare with PyTorch result
        result.backward(np.array([1, 1, 1]))
        result_torch.backward(torch.tensor([1, 1, 1]))

        np.testing.assert_almost_equal(
            tensor1.grad,
            tensor1_torch.grad.detach().numpy(),
            decimal=2
        )
        np.testing.assert_almost_equal(
            tensor2.grad,
            tensor2_torch.grad.detach().numpy(),
            decimal=2
        )

    def test_rtruediv(self):
        # Test truediv method
        tensor = Tensor([1, 2, 3], requires_grad=True)
        b = 3

        result = b / tensor

        tensor_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        result_torch = b / tensor_torch

        np.testing.assert_almost_equal(
            result.data,
            result_torch.data.detach().numpy(),
            decimal=2
        )

        result.backward(np.ones_like(result.data))
        result_torch.backward(torch.tensor([1, 1, 1]))

        np.testing.assert_almost_equal(
            tensor.grad,
            tensor_torch.grad.detach().numpy(),
            decimal=2
        )

    def test_backward_scalar(self):
        t = Tensor(2.0, requires_grad=True)
        t.backward()
        self.assertEqual(t.grad, 1.0)

    def test_backward_vector(self):
        t = Tensor([1., 2., 3.], requires_grad=True)
        grad = np.ones_like(t.data)
        t.backward(grad)
        np.testing.assert_array_equal(t.grad, np.array([1., 1., 1.]))

    def test_backward_with_graph(self):
        x = Tensor(2.0, requires_grad=True)
        x.backward()
        self.assertEqual(x.grad, 1.0)

    def test_backward_error(self):
        t = Tensor([1., 2., 3.], requires_grad=True)
        with self.assertRaises(ValueError):
            t.backward()

    def test_backward(self):
        # Test backward method
        tensor = Tensor(np.array([1, 2, 3]), requires_grad=True)
        tensor.backward(np.array([1, 1, 1]))
        self.assertEqual(tensor.grad.tolist(), [1.0, 1.0, 1.0])

        # Compare with PyTorch result
        tensor_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        tensor_torch.backward(torch.tensor([1, 1, 1]))
        self.assertEqual(tensor.grad.tolist(), tensor_torch.grad.tolist())

    def test_backward_long_chain(self):
        a_data1 = np.array([1, 0, 1], dtype=np.float32)
        b_data1 = np.array([1, 1, 0], dtype=np.float32)

        a_data2 = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ], dtype=np.float32)
        b_data2 = np.array([
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
        ], dtype=np.float32)

        a_data3 = np.random.randn(5, 5)
        b_data3 = np.random.randn(1, 5)

        data = [
            (a_data1, b_data1),
            (a_data2, b_data2),
            (a_data3, b_data3)
        ]

        def runner(a_data: np.ndarray, b_data: np.ndarray):
            a = Tensor(a_data, requires_grad=True)
            b = Tensor(b_data, requires_grad=True)

            a_torch = torch.tensor(a_data, requires_grad=True)
            b_torch = torch.tensor(b_data, requires_grad=True)

            def network(a, b):
                action1 = ((a @ b.T + b)**2 - (a @ b.T + a).log()) / 2
                action2 = (-a + b * 3).tanh()
                action3 = action1.sum(axis = None, keepdims = False) + action2.mean()
                action4 = (-a - b + a @ b.T).abs()

                action5 = action1.sum(axis=0, keepdims=True)

                result = action1 + action2 + action3 + action4 + action5
                return result
            
            res = network(a, b)
            res_torch = network(a_torch, b_torch)

            np.testing.assert_allclose(res.data, res_torch.detach().numpy())

            res.backward(np.ones_like(res.data))
            res_torch.backward(torch.ones_like(res_torch))

            np.testing.assert_almost_equal(
                a.grad,
                a_torch.grad.detach().numpy(),
                decimal=2
            )

            np.testing.assert_almost_equal(
                b.grad,
                b_torch.grad.detach().numpy(),
                decimal=2
            )

        for a, b in data:
            runner(a, b)



if __name__ == '__main__':
    unittest.main()