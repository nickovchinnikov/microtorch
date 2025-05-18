import unittest
from unittest.mock import Mock

import numpy as np

from src.tensor.ops.decorators import auto_backend
from src.tensor.tensor import Tensor
from src.tensor.backend import Backend
from src.tensor.backend.types import Device


class TestAutoBackend(unittest.TestCase):
    def setUp(self):
        # Dummy tensor for inference tests
        self.tensor = Tensor(np.array([1.0, 2.0]), device=Device.CPU)

    def test_auto_backend_explicit_backend(self):
        # Should use provided backend, not infer from args
        mock_backend = Mock(spec=Backend)

        @auto_backend
        def op(tensor, backend=None):
            return backend  # expose backend used

        result = op(self.tensor, backend=mock_backend)
        self.assertEqual(result, mock_backend)

    def test_auto_backend_infers_backend_from_tensor(self):
        @auto_backend
        def op(tensor, backend=None):
            return backend.__class__.__name__  # Sanity check

        backend_name = op(self.tensor)
        self.assertIsInstance(backend_name, str)
        self.assertIn("Backend", backend_name)

    def test_auto_backend_raises_when_no_tensorlike_found(self):
        @auto_backend
        def op(x, y, backend=None):
            return backend

        with self.assertRaises(ValueError) as cm:
            op("not a tensor", 123)

        self.assertIn("No tensor with a device", str(cm.exception))
