import unittest

from src.nn.module import Module
from src.nn.param import Parameter
from src.nn.sequential import Sequential
from src.tensor.tensor import Tensor


class MockModule(Module):
    def __init__(self, param_value):
        super().__init__()
        self.param = Parameter(param_value)

    def forward(self, x):
        return x + self.param.data


class TestSequential(unittest.TestCase):
    def test_init(self):
        modules = [MockModule(1), MockModule(2), MockModule(3)]
        sequential = Sequential(*modules)
        self.assertEqual(len(sequential.modules), 3)

    def test_parameters(self):
        modules = [MockModule(1), MockModule(2), MockModule(3)]
        sequential = Sequential(*modules)
        params = list(sequential.parameters())
        self.assertEqual(len(params), 3)
        self.assertTrue(all(isinstance(p, Parameter) for p in params))

    def test_forward(self):
        modules = [MockModule(1), MockModule(3)]
        sequential = Sequential(*modules)
        input_tensor = Tensor([1, 2, 3])
        output = sequential(input_tensor)
        self.assertEqual(len(output.data), 3)

    def test_empty_sequential(self):
        sequential = Sequential()
        self.assertEqual(len(sequential.modules), 0)
        self.assertEqual(len(list(sequential.parameters())), 0)

        input_tensor = Tensor([5])
        output = sequential(input_tensor)
        self.assertEqual(output.data, [5])  # No changes to input


if __name__ == "__main__":
    unittest.main()
