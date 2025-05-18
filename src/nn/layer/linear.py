from src.nn.module import Module
from src.nn.param import Parameter
from src.tensor.tensor import Tensor


class Linear(Module):
    r"""
    A linear layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        r"""
        Initialize the linear layer.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            out_features, in_features, init_method="uniform", gain=.1
        )
        self.bias = Parameter(
            out_features, init_method="uniform", gain=.1
        ) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Forward pass.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        # Check dimensions of input tensors
        assert x.ndim in (2, 3), f"Input must be 2D or 3D Tensor! x.ndim={x.ndim}"

        # Check if the last dimension of input matches in_features
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Last dimension of input: {x.shape[-1]}"
                f"does not match in_features: {self.in_features}"
            )

        # Transpose axes for matrix multiplication if input is 3D
        transpose_axes = (0, 2, 1) if x.ndim == 3 else None

        # 3d: (batch_size, 1, in_features) @ (batch_size, in_features, 1) 
        # => (batch_size, 1, 1)
        # 2d: (batch_size, in_features) @ (batch_size, in_features) 
        # => (batch_size)
        output = self.weight @ x.transpose(transpose_axes)

        # 3d: (batch_size, 1, 1)
        # 2d: (batch_size)
        output = output.transpose(transpose_axes)

        if self.bias is not None:
            # Reshape bias tensor to match output of matrix multiplication
            if output.data.ndim == 2:
                bias = self.bias.unsqueeze(dim=0)
            else:
                bias = self.bias.unsqueeze(dim=0).unsqueeze(axis=0)

            # output = output + bias
            output += bias

        return output
