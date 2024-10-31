# MicroTorch

A lightweight **automatic differentiation library** built on top of `NumPy`. This project enables easy tensor manipulations, gradients computation, and custom-defined layers for neural networks.

![MicroTorch Logo](./microtorch.png)

## Features

- **Automatic Differentiation**: Supports autograd for backpropagation.
- **Flexible Operations**: Provides a variety of tensor operations like addition, multiplication, transpose, reshape, and more, with gradient support.

## Installation

To use this library, install the necessary packages:

```bash
pip install numpy
```

## Spiral Dataset MLP Demo

### [Jupyter Notebook Demo](./demo.ipynb)

This notebook demonstrates the implementation of a **Multilayer Perceptron (MLP)** using the `microtorch` library to classify a synthetic spiral dataset. The MLP model consists of a customizable architecture that learns to distinguish between two intertwined spirals, with each layer followed by a non-linear activation function.

### Key Components
1. **Dataset Creation**: A custom `make_spiral_dataset` function generates a synthetic dataset where two classes form spiral patterns.
2. **Model Architecture**: The `MLP` class initializes a Sequential model with Linear layers and Tanh activation functions.
3. **Training Process**:
   - Forward pass through the model.
   - Computation of Mean Squared Error (MSE) loss, which is used as an example but can be replaced with Cross-Entropy or Binary Cross-Entropy losses.
   - Backpropagation and optimization using Stochastic Gradient Descent (SGD).
4. **Visualization**: After training, a decision boundary is plotted alongside the dataset, showing the model's classification regions.

### Requirements
- `numpy`
- `matplotlib`
- `microtorch`


## Usage Examples

### Basic Tensor Initialization

```python
from my_tensor import Tensor

# Create a tensor with random data, requiring gradient computation
x = Tensor.randn((3, 3), require_grad=True)
y = Tensor.randn((3, 3), require_grad=True)
```

### Forward and Backward Computation

```python
# Perform operations
z = x * y + x**2 - y

# Compute gradients
z.backward()

# Access gradient
print(x.grad)
print(y.grad)
```

### Reshape, Transpose, and More Operations

```python
# Reshape tensor
reshaped = x.view((9,))

# Transpose tensor
transposed = x.T

# Matrix multiplication
result = x @ y
result.backward()
```

### More Examples

```python
# Sum, mean, and power functions
sum_result = x.sum()
mean_result = x.mean()
pow_result = x**2

# Logarithmic and hyperbolic functions
log_result = x.log()
tanh_result = x.tanh()

# Squeeze and unsqueeze
squeezed = x.squeeze()
unsqueezed = x.unsqueeze()
```

## `nn` Package in `microtorch`

The `nn` package in `microtorch` provides core modules for constructing neural network models, including layers, activations, optimizers, and loss functions. This package is designed to be modular and extensible, allowing users to create, train, and evaluate custom neural networks with ease.

### Key Modules

- **`Module`**: The base class for all layers and neural network components. Each custom layer, activation, or other neural component should inherit from this class. `Module` provides:
  - `forward`: The method to define computations of each layer.
  - `parameters`: Returns all model parameters for optimization.
  - `train` and `eval`: Sets the layer's mode for training or evaluation.
  - `zero_grad`: Clears gradients of all parameters.

- **`Parameter`**: A subclass of `Tensor` representing model parameters. Parameters can be initialized using various methods, such as **Xavier**, **He**, **normal**, or **uniform** distributions, making them suitable for different types of neural network architectures.

- **Layers**:
  - **`Linear`**: Implements a fully connected (dense) layer with learnable weights and optional biases. Handles both 2D and 3D input tensors for versatility.
  - **`Sequential`**: A container to stack layers sequentially, simplifying network building.

- **Activations**:
  - **`Tanh`**: Implements the hyperbolic tangent (tanh) activation function, adding non-linearity to the model.

- **Loss Functions**:
  - **`Loss`**: A base class for all loss functions, including a `reduction_loss` method to handle different types of reduction (mean, sum, or none).
  - **`MSELoss`**: Computes mean squared error, commonly used for regression tasks.
  - **`L1Loss`**: Calculates the mean absolute error.
  - **`CrossEntropyLoss`**: Implements the cross-entropy loss for multi-class classification tasks.
  - **`BCELoss`**: Computes binary cross-entropy loss, suitable for binary classification tasks.

- **Optimizers**:
  - **`Optimizer`**: The base class for optimizers, defining a standard interface for updating model parameters.
  - **`SGD`**: Implements Stochastic Gradient Descent (SGD) for updating parameters based on gradients.

### Example Usage

Below is an example demonstrating how to define and use a model with the `Linear` and `Tanh` layers, apply a loss function, and optimize using SGD.

```python
from src.nn import Module, Linear, Tanh, MSELoss
from src.nn.optimizer import SGD
from src.tensor import Tensor

class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.layer = Linear(2, 1)
        self.activation = Tanh()

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer(x)
        return self.activation(x)

model = SimpleModel()
optimizer = SGD(model.parameters, lr=0.01)
loss_fn = MSELoss()

# Forward pass, loss calculation, and backpropagation
prediction = model(Tensor([0.5, -0.2]))
loss = loss_fn(prediction, Tensor([1.0]))
loss.backward()
optimizer.step()
```

This package enables users to create and train neural networks with customizable architectures, activation functions, and loss functions.
