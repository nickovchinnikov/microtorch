import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from src.tensor.backend.dispatch import (
    device_cast,
    get_backend,
    get_dtype,
    check_cuda,
)
from src.tensor.backend.numpy import NumpyBackend
from src.tensor.backend.types import Device


class TestBackendRegistry(unittest.TestCase):

    def test_device_cast_from_string(self):
        result = device_cast("cpu")
        self.assertEqual(result, Device.CPU)

    def test_device_cast_from_enum(self):
        result = device_cast(Device.CPU)
        self.assertEqual(result, Device.CPU)

    def test_get_backend_valid(self):
        backend = get_backend(Device.CPU)
        self.assertIsInstance(backend, NumpyBackend)

    def test_get_backend_invalid_raises(self):
        # Patch dispatch._backends to simulate unregistered device
        import src.tensor.backend.dispatch as dispatch

        with patch.object(dispatch, "_backends", {}):
            with self.assertRaises(ValueError) as cm:
                get_backend(Device.CPU)

            self.assertIn("not registered or supported", str(cm.exception))

    def test_check_cuda_importerror(self):
        # Simulate CuPy not installed by removing it from sys.modules
        with patch.dict("sys.modules", {"cupy": None}):
            result = check_cuda()
            self.assertFalse(result)

    def test_check_cuda_available(self):
        # Simulate CuPy installed and CUDA available
        mock_cp = MagicMock()
        mock_cp.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"cupy": mock_cp}):
            result = check_cuda()
            self.assertTrue(result)

    def test_get_dtype_cpu_float32(self):
        dtype = get_dtype(Device.CPU, np.float32)
        self.assertEqual(dtype, np.float32)
