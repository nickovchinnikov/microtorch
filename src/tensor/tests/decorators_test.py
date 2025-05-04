import unittest

import numpy as np

from src.tensor.device import Device, DType
from src.tensor.tensor import (
    Tensor,
    data_cast,
    data_gate,
    device_gate,
    from_op,
    input_gate,
    is_tensorlike,
    op_gate,
)
from src.tensor.types import TProps


class TestDecorators(unittest.TestCase):

    def test_is_tensorlike(self):
        self.assertTrue(is_tensorlike(Tensor([1.0])))
        self.assertFalse(is_tensorlike(42))

    def test_data_cast_tensor_passthrough(self):
        t = Tensor([1.0])
        self.assertIs(data_cast(t), t)

    def test_data_cast_from_list(self):
        t = data_cast([1.0, 2.0])
        self.assertIsInstance(t, Tensor)
        np.testing.assert_array_equal(t.data, [1.0, 2.0])

    def test_data_gate_casts_input(self):
        @data_gate
        def add(self: Tensor, other: Tensor):
            return self.data + other.data

        a = Tensor([1.0])
        result = add(a, [2.0])
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_device_gate_same_device(self):
        @device_gate
        def add(self: Tensor, other: Tensor):
            return self.data + other.data

        a = Tensor([1.0], device=Device.CPU)
        b = Tensor([2.0], device=Device.CPU)
        result = add(a, b)
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_device_gate_raises_on_mismatch(self):
        @device_gate
        def add(self: Tensor, other: Tensor):
            return self.data + other.data

        a = Tensor([1.0], device=Device.CPU)
        b = Tensor([2.0], device=Device.CPU)
        b.device = Device.CUDA  # Force mismatch manually
        with self.assertRaises(ValueError):
            add(a, b)

    def test_input_gate_combines_data_and_device(self):
        @input_gate
        def subtract(self: Tensor, other: Tensor):
            return self.data - other.data

        a = Tensor([5.0])
        result = subtract(a, [2.0])
        np.testing.assert_array_equal(result, np.array([3.0]))

    def test_from_op_wraps_tprops(self):
        @from_op
        def make_tensor():
            return TProps(
                _data=np.array([1.0]),
                requires_grad=False,
                dependencies=[],
                device=Device.CPU,
                dtype=DType.FLOAT32,
            )

        t = make_tensor()
        self.assertIsInstance(t, Tensor)
        np.testing.assert_array_equal(t.data, [1.0])

    def test_op_gate_combines_all(self):
        @op_gate
        def add_op(self: Tensor, other: Tensor):
            return TProps(
                _data=self.data + other.data,
                requires_grad=self.requires_grad or other.requires_grad,
                dependencies=[],
                device=self.device,
                dtype=self.dtype,
            )

        a = Tensor([1.0])
        result = add_op(a, [2.0])
        self.assertIsInstance(result, Tensor)
        np.testing.assert_array_equal(result.data, [3.0])


if __name__ == "__main__":
    unittest.main()
