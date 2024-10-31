import unittest

import numpy as np

from src.nn.module import Module
from src.nn.param import Parameter
from src.tensor import Tensor


class TestModule(unittest.TestCase):
    class SimpleModule(Module):
        def __init__(self):
            super().__init__()
            self.param1 = Parameter(3)
            self.param2 = Parameter(3)
            self.submodule = Module()
            self.submodule.param = Parameter(3)

        def forward(self, x):
            return x + self.param1.data + self.param2.data

    def test_init(self):
        module = Module()
        self.assertTrue(module.train_mode)

    def test_call(self):
        module = self.SimpleModule()
        input_tensor = Tensor(np.array([1.0, 1.0, 1.0]))
        output = module(input_tensor)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, (3,))

    def test_forward_not_implemented(self):
        module = Module()
        with self.assertRaises(NotImplementedError):
            module.forward(Tensor(np.array([1.0])))

    def test_train(self):
        module = self.SimpleModule()
        module.eval()
        module.train()
        self.assertTrue(module.train_mode)
        self.assertTrue(module.submodule.train_mode)

    def test_eval(self):
        module = self.SimpleModule()
        module.eval()
        self.assertFalse(module.train_mode)
        self.assertFalse(module.submodule.train_mode)

    def test_parameters(self):
        module = self.SimpleModule()
        params = list(module.parameters())
        self.assertEqual(len(params), 3)
        self.assertIsInstance(params[0], Parameter)
        self.assertEqual(params[0].shape, (3,))
        self.assertEqual(params[1].shape, (3,))
        self.assertEqual(params[2].shape, (3,))

    def test_zero_grad(self):
        module = self.SimpleModule()
        for param in module.parameters():
            param.grad = np.array([1.0, 1.0, 1.0])
        module.zero_grad()
        for param in module.parameters():
            np.testing.assert_array_equal(param.grad, np.array([0.0, 0.0, 0.0]))

    def test_params_count(self):
        module = self.SimpleModule()
        self.assertEqual(module.params_count(), 9)  # 3 parameters with 3 elements each


if __name__ == "__main__":
    unittest.main()
