import unittest
from unittest.mock import Mock

import numpy as np

from src.tensor.backend import Device, DType
from src.tensor.tensor import Tensor
from src.tensor.types import Leaf, TensorLike, TProps


class TestTypes(unittest.TestCase):
    def test_leaf_structure(self):
        dummy_tensor = Mock(spec=TensorLike)
        dummy_fn = lambda x: x + 1
        leaf = Leaf(value=dummy_tensor, grad_fn=dummy_fn)
        self.assertIs(leaf.value, dummy_tensor)
        self.assertEqual(leaf.grad_fn(np.array([2])), np.array([3]))

    def test_tprops_props(self):
        data = np.ones((2, 2))
        tprops = TProps(
            _data=data,
            requires_grad=True,
            dependencies=[],
            device=Device.CPU,
            dtype=DType.float32
        )
        self.assertEqual(tprops.props(), (data, True, [], Device.CPU, DType.float32))

    def test_tensorlike_conformance(self):
        tensor = Tensor([[1, 2]], requires_grad=False)
        self.assertIsInstance(tensor, TensorLike)

