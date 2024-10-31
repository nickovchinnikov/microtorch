from typing import Literal

from src.nn.module import Module
from src.tensor import Tensor


def reduction_loss(
    loss: Tensor, reduction: Literal["mean", "sum", "none"] = "mean"
) -> Tensor:
    r"""
    Reduction loss function.
    """

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class Loss(Module):
    r"""
    Base class for loss functions.
    """

    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean"):
        self.reduction = reduction

    def compute_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        r"""
        Compute loss function.
        """
        raise NotImplementedError

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        r"""
        Forward pass.
        Apply the reduction to the loss.
        """
        assert (
            prediction.shape == target.shape
        ), "Input and target must have the same shape"
        loss = self.compute_loss(prediction, target)
        return reduction_loss(loss, self.reduction)


class L1Loss(Loss):
    r"""
    L1 loss function.
    """

    def __init__(self):
        super().__init__(reduction="mean")

    def compute_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        r"""
        Compute L1 loss function.
        """

        return (prediction - target).abs()


class MSELoss(Loss):
    r"""
    Mean Squared Loss function.
    """

    def __init__(self):
        super().__init__(reduction="mean")

    def compute_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        r"""
        Compute Mean Squared Loss function.
        """

        return (prediction - target).pow(2)
    
class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__(reduction="mean")    

    def compute_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        return -(target - prediction.log())


class BCELoss(Loss):
    r"""
    Binary Cross Entropy Loss function.
    """

    def __init__(self):
        super().__init__(reduction="mean")

    def compute_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        r"""
        Compute Binary Cross Entropy Loss function.
        """

        return -(target * prediction.log() + (1 - target) * (1 - prediction).log())
