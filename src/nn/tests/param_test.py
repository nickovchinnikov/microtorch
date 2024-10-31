import unittest

import numpy as np

from src.nn.param import Parameter


class TestParameter(unittest.TestCase):
    def test_init_with_data(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        param = Parameter(2, 2, data=data)
        np.testing.assert_array_equal(param.data, data)
        self.assertTrue(param.requires_grad)

    def test_init_without_data(self):
        param = Parameter(2, 3)
        self.assertEqual(param.shape, (2, 3))
        self.assertTrue(param.requires_grad)

    def test_xavier_init(self):
        param = Parameter(100, 100, init_method='xavier')
        self.assertAlmostEqual(np.mean(param.data), 0, places=2)
        self.assertAlmostEqual(np.std(param.data), np.sqrt(2.0 / 200), places=2)

    def test_he_init(self):
        param = Parameter(100, 100, init_method='he')
        self.assertAlmostEqual(np.mean(param.data), 0, places=2)
        self.assertAlmostEqual(np.std(param.data), np.sqrt(2.0 / 100), places=2)

    def test_normal_init(self):
        param = Parameter(100, 100, init_method='normal')
        self.assertAlmostEqual(np.mean(param.data), 0, places=1)
        self.assertAlmostEqual(np.std(param.data), 1, places=1)

    def test_invalid_init_method(self):
        with self.assertRaises(ValueError):
            Parameter(2, 2, init_method='invalid')

    def test_gain(self):
        gain = 2.0
        param = Parameter(100, 100, init_method='xavier', gain=gain)
        expected_std = gain * np.sqrt(2.0 / 200)
        self.assertAlmostEqual(np.std(param.data), expected_std, places=2)

if __name__ == '__main__':
    unittest.main()