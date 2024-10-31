from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

type Scalar = Union[int, float]
type Data = Union[Scalar, list, np.ndarray, "Tensor"]


@dataclass(frozen=True)
class Leaf:
    value: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor:
    def __init__(
        self,
        data: Data,
        requires_grad: bool = False,
        dependencies: Optional[List[Leaf]] = [],
        dtype=np.float32
    ) -> None:
        self._data = Tensor.build_ndarray(data, dtype)
        self.dtype = self._data.dtype

        self.requires_grad = requires_grad
        self.dependencies: List[Leaf] = dependencies

        self.grad: np.ndarray = None

        if self.requires_grad:
            self.zero_grad()
    
    @property
    def ndim(self) -> int:
        return self._data.ndim
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape
    
    @property
    def size(self) -> int:
        return self.data.size

    @property
    def data(self) -> np.ndarray:
        """Return the data of the tensor"""
        return self._data
    
    def zero_grad(self) -> None:
        r"""
        Zero the gradients of all parameters
        """
        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=float)
        else:
            self.grad.fill(0.0)

    @data.setter
    def data(self, new_data: Data) -> None:
        """Set the data of the tensor"""
        self._data = Tensor.build_ndarray(new_data)
        self.zero_grad()

    @staticmethod
    def build_ndarray(
        data: Data,
        dtype = np.float32
    ) -> np.ndarray:
        # TODO: You can use CUDA here to wrap instead of np cupy !
        # Case of, data is a np.ndarray
        if isinstance(data, np.ndarray):
            # Ensure np.ndarray includes floats
            types = (np.float32, np.float64, np.int16, np.int32)
            if data.dtype in types:
                return data
            # Cast data
            return data.astype(dtype)
        # Case of, data is a tensor
        if isinstance(data, Tensor):
            return np.array(data.data, dtype=dtype)
        # Case of, data is a list, float or int
        return np.array(data, dtype=dtype)
    
    @staticmethod
    def data_gate(
        data_object: Data,
    ) -> "Tensor":
        # Check if the other object has all the required attributes
        required_attrs = [
            "_data",
            "shape",
            "dtype",
            "dependencies",
            "requires_grad",
            "grad",
        ]

        if all(hasattr(data_object, attr) for attr in required_attrs):
            return data_object
        else:
            return Tensor(data_object)

    @staticmethod
    def randn(dims: Tuple[int] | int = (), require_grad=False):
        if type(dims) is int:
            dims = (dims,)
        return Tensor(np.random.randn(*dims), require_grad)

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad}, shape={self.shape})"

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        assert self.requires_grad, "Backward was called on a non-required-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor.build_ndarray(1.0)
            else:
                raise ValueError("Grad must be provided if tensor has shape")

        self.grad = self.grad + grad

        for dependency in self.dependencies:
            backward_grad = dependency.grad_fn(grad)
            dependency.value.backward(backward_grad)

    def view(self, shape: Tuple[int, ...], stride: Tuple[int, ...] = None) -> "Tensor":
        r"""
        Returns a new Tensor object with the same underlying data, but with a different shape.

        Args:
            shape (Tuple[int, ...]): The new shape of the Tensor.
            stride (Tuple[int, ...]): The stride of the new Tensor.
                If not specified, the stride will be calculated automatically.

        Returns:
            A new Tensor object with the same underlying data, but with the specified shape.
        """

        if stride is None:
            stride = self.data.strides

        output: np.ndarray = self.data.reshape(shape)
        dependencies: List[Leaf] = []

        if self.requires_grad:
            def _bkwd(grad: np.ndarray) -> np.ndarray:
                return grad.reshape(self.shape)

            dependencies.append(Leaf(value=self, grad_fn=_bkwd))

        return Tensor(output, self.requires_grad, dependencies, dtype=self.dtype)

    def transpose(self, axes: Tuple[int, ...] = None) -> "Tensor":
        output = np.transpose(self.data, axes=axes)
        dependencies: List[Leaf] = []

        def _bkwd(grad: np.ndarray) -> np.ndarray:
            return np.transpose(grad, axes=axes)

        if self.requires_grad:
            dependencies.append(
                Leaf(value=self, grad_fn=_bkwd)
            )

        return Tensor(output, self.requires_grad, dependencies)
    
    @property
    def T(self) -> "Tensor":
        return self.transpose()
    
    def squeeze(self, dim: int | Tuple[int] = 0) -> "Tensor":
        output = np.squeeze(self.data, axis=dim)
        dependencies: List[Leaf] = []

        def _bkwd(grad: np.ndarray) -> np.ndarray:
            return np.expand_dims(grad, axis=dim)

        if self.requires_grad:
            dependencies.append(
                Leaf(value=self, grad_fn=_bkwd)
            )

        return Tensor(output, self.requires_grad, dependencies)
    
    def unsqueeze(self, dim: Tuple[int] = 0) -> "Tensor":
        output = np.expand_dims(self.data, axis=dim)
        dependencies: List[Leaf] = []

        def _bkwd(grad: np.ndarray) -> np.ndarray:
            return np.squeeze(grad, axis=dim)

        if self.requires_grad:
            dependencies.append(
                Leaf(
                    value=self,
                    grad_fn=_bkwd
                )
            )

        return Tensor(output, self.requires_grad, dependencies)
    
    def sum(self, axis: int = None, keepdims: bool = False) -> "Tensor":
        output = self.data.sum(axis=axis, keepdims=keepdims)
        dependencies: List[Leaf] = []

        def _bkwd(grad: np.ndarray) -> np.ndarray:
            if keepdims:
                expanded_grad = np.expand_dims(grad, axis=axis)
                ones = np.ones_like(expanded_grad)
                grad = expanded_grad * ones
            return np.sum(grad, axis=axis)

        if self.requires_grad:
            dependencies.append(
                Leaf(
                    value=self,
                    grad_fn=_bkwd
                )
            )

        return Tensor(output, self.requires_grad, dependencies)
    
    def mean(self) -> "Tensor":
        return self.sum() / self.size

    def abs(self) -> "Tensor":
        output = np.abs(self.data)
        dependencies: List[Leaf] = []

        def _bkwd(grad: np.ndarray) -> np.ndarray:
            return grad * np.sign(self.data)

        if self.requires_grad:
            dependencies.append(
                Leaf(
                    value=self,
                    grad_fn=_bkwd
                )
            )

        return Tensor(output, self.requires_grad, dependencies)
    
    def log(self) -> "Tensor":
        output = np.log(self.data)
        dependencies: List[Leaf] = []

        def _bkwd(grad: np.ndarray) -> np.ndarray:
            return grad / self.data

        if self.requires_grad:
            dependencies.append(
                Leaf(
                    value=self,
                    grad_fn=_bkwd
                )
            )

        return Tensor(output, self.requires_grad, dependencies)
    
    def tanh(self) -> "Tensor":
        output = np.tanh(self.data)
        dependencies: List[Leaf] = []

        def _bkwd(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - output**2)

        if self.requires_grad:
            dependencies.append(
                Leaf(
                    value=self,
                    grad_fn=_bkwd
                )
            )

        return Tensor(output, self.requires_grad, dependencies)
    
    def pow(self, pow: Scalar) -> "Tensor":
        # Perform power operation
        output = self.data**pow
        dependencies: List[Leaf] = []

        def _bkwd(grad: np.ndarray) -> np.ndarray:
            return grad * (pow * (self.data**(pow-1)))

        if self.requires_grad:
            dependencies.append(
                Leaf(
                    value=self,
                    grad_fn=_bkwd
                )
            )

        return Tensor(output, self.requires_grad, dependencies)

    ###########################################################################
    ############################## Operator Overload ##########################
    ###########################################################################
  
    def __getitem__(self, index: Union["Tensor", np.ndarray]) -> "Tensor":
        r"""
        Tensor indexing operation.

        Args:
            index (Union["Tensor", np.ndarray]): The index to select from the tensor.

        Returns:
            Tensor: The selected tensor.
        """

        index = Tensor.data_gate(index).data
        output = self.data[index]
        dependencies = []

        if self.requires_grad:
            def _bkwd(grad):
                r"""
                Backward pass for tensor indexing.
                """
                full_grad = np.zeros_like(self.data)
                full_grad[index] = grad
                return full_grad
            
            dependencies.append(Leaf(value=self, grad_fn=_bkwd))
            
        return Tensor(output, self.requires_grad, dependencies)
    
    @staticmethod
    def _bkwd_broadcast(tensor: "Tensor"):
        r"""
        Sum across dimensions for broadcast.
        Backward closure function for grad.
        """

        def _bkwd(grad: np.ndarray) -> np.ndarray:
            dimensions = grad.ndim - tensor.ndim
            # Sum across broadcasted dimensions
            for _ in range(dimensions):
                grad = grad.sum(axis=0, keepdims=True)
            # Sum across singleton dimensions
            for index, dimension in enumerate(tensor.shape):
                if dimension == 1:
                    grad = grad.sum(axis=index, keepdims=True)
            return grad

        return _bkwd

    @staticmethod
    def _add(a: "Tensor", b: "Tensor") -> "Tensor":
        output = a.data + b.data
        requires_grad = a.requires_grad or b.requires_grad
        dependencies: List[Leaf] = []

        if a.requires_grad:
            dependencies.append(
                Leaf(value=a, grad_fn=Tensor._bkwd_broadcast(a))
            )

        if b.requires_grad:
            dependencies.append(
                Leaf(value=b, grad_fn=Tensor._bkwd_broadcast(b))
            )

        return Tensor(output, requires_grad, dependencies)

    def __add__(self, other: Data) -> "Tensor":
        return Tensor._add(self, Tensor.data_gate(other))
    
    def __radd__(self, other: Data) -> "Tensor":
        return Tensor._add(Tensor.data_gate(other), self)
    
    def __iadd__(self, other: Data) -> "Tensor":
        r"""
        In-place addition self: += other
        There is no gradient function for in-place operations!
        """
        self.data = self.data + Tensor.build_ndarray(other)
        return self
    
    def __sub__(self, other: Data) -> "Tensor":
        return self + (-Tensor.data_gate(other))

    def __rsub__(self, other: Data) -> "Tensor":
        return Tensor.data_gate(other) + (-self)
    
    def __isub__(self, other: Data) -> "Tensor":
        r"""
        In-place subtraction self: -= other
        There is no gradient function for in-place operations!
        """
        self.data = self.data - Tensor.build_ndarray(other)
        return self
    
    def __neg__(self) -> "Tensor":
        output = -self.data
        dependencies: List[Leaf] = []

        if self.requires_grad:
            dependencies.append(
                Leaf(value=self, grad_fn=lambda grad: -grad)
            )

        return Tensor(output, self.requires_grad, dependencies)
    
    @staticmethod
    def mul(a: "Tensor", b: "Tensor") -> "Tensor":
        a = Tensor.data_gate(a)
        b = Tensor.data_gate(b)

        output = a.data * b.data
        requires_grad = a.requires_grad or b.requires_grad
        dependencies: List[Leaf] = []

        def _backward(a: Tensor, b: Tensor):
            r"""
            Backward closure function for Mul.
            """

            def _bkwd(grad: np.ndarray) -> np.ndarray:
                r"""
                Backward gradient function for Mul.
                """

                # Multiply grad by tensor b data
                grad = grad * b.data
                # Reduce grad to the correct shape
                return Tensor._bkwd_broadcast(a)(grad)

            return _bkwd

        if a.requires_grad:
            dependencies.append(
                Leaf(
                    value=a,
                    grad_fn=_backward(a, b)
                )
            )

        if b.requires_grad:
            dependencies.append(
                Leaf(
                    value=b,
                    grad_fn=_backward(b, a)
                )
            )

        return Tensor(output, requires_grad, dependencies)

    def __mul__(self, other: Data) -> "Tensor":
        return Tensor.mul(self, Tensor.data_gate(other))
    
    def __rmul__(self, other: Data) -> "Tensor":
        return Tensor.mul(Tensor.data_gate(other), self)
    
    def __imul__(self, other: Data) -> "Tensor":
        r"""
        In-place multiplication self: *= other
        There is no gradient function for in-place operations!
        """
        self.data = self.data * Tensor.build_ndarray(other)
        return self
    
    @staticmethod
    def _matmul(a: "Tensor", b: "Tensor") -> "Tensor":
        output = a.data @ b.data
        requires_grad = a.requires_grad or b.requires_grad
        dependencies: List[Leaf] = []

        if a.requires_grad:
            def _bkwd(grad: np.ndarray) -> np.ndarray:
                r"""
                Backward gradient function for MatMul with respect to a.
                """
                if b.ndim > 1:
                    return grad @ b.data.swapaxes(-1, -2)
                return np.outer(grad, b.data.T).squeeze()
    
            dependencies.append(
                Leaf(
                    value=a,
                    grad_fn=_bkwd
                )
            )

        if b.requires_grad:
            def _bkwd(grad: np.ndarray) -> np.ndarray:
                r"""
                Backward gradient function for MatMul with respect to b.
                """
                if a.ndim > 1:
                    return a.data.swapaxes(-1, -2) @ grad
                return np.outer(a.data.T, grad).squeeze()
            
            dependencies.append(
                Leaf(
                    value=b,
                    grad_fn=_bkwd
                )
            )

        return Tensor(output, requires_grad, dependencies)
    
    def __matmul__(self, other: Data) -> "Tensor":
        return Tensor._matmul(self, Tensor.data_gate(other))
    
    def __rmatmul__(self, other: Data) -> "Tensor":
        return Tensor._matmul(Tensor.data_gate(other), self)
    
    def __pow__(self, pow: Scalar) -> "Tensor":
        return self.pow(pow)
    
    def __truediv__(self, other: Data) -> "Tensor":
        other = Tensor.data_gate(other)
        return self * (other**-1)

    def __rtruediv__(self, other: Data) -> "Tensor":
        other = Tensor.data_gate(other)
        return other * (self**-1)
    
    def __itruediv__(self, other: Data) -> "Tensor":
        r"""
        In-place division self: /= other
        There is no gradient function for in-place operations!
        """
        self.data = self.data / Tensor.build_ndarray(other)
        return self
