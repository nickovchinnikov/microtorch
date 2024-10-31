import unittest

import numpy as np

from src.nn.loss import BCELoss, L1Loss, MSELoss
from src.tensor import Tensor


class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        self.prediction = Tensor(np.array([0.1, 0.2, 0.3, 0.4]))
        self.target = Tensor(np.array([0.0, 0.0, 1.0, 1.0]))

    def test_l1_loss(self):
        loss_fn = L1Loss()
        loss = loss_fn(self.prediction, self.target)
        expected_loss = np.mean(np.abs(self.prediction.data - self.target.data))
        self.assertAlmostEqual(loss.data, expected_loss, places=6)

    def test_mse_loss(self):
        loss_fn = MSELoss()
        loss = loss_fn(self.prediction, self.target)
        expected_loss = np.mean(np.square(self.prediction.data - self.target.data))
        self.assertAlmostEqual(loss.data, expected_loss, places=6)

    def test_bce_loss(self):
        loss_fn = BCELoss()
        loss = loss_fn(self.prediction, self.target)
        expected_loss = -np.mean(
            self.target.data * np.log(self.prediction.data)
            + (1 - self.target.data) * np.log(1 - self.prediction.data)
        )
        self.assertAlmostEqual(loss.data, expected_loss, places=6)


if __name__ == "__main__":
    unittest.main()
