import pickle
from functools import wraps
from pathlib import Path
from typing import TypeGuard, final

from .backend import (
    Backend,
    Device,
    DType,
    Scalar,
    Vector,
    device_cast,
    get_backend,
    get_dtype,
)
from .ops import Ops
from .types import (
    Data,
    DependenciesList,
    Dims,
    Index,
    Shape,
    TensorLike,
    TProps,
)


def is_tensorlike(obj: object) -> TypeGuard[TensorLike]:
    return isinstance(obj, Tensor)

def data_cast(t: Data | "Tensor", device: Device) -> "Tensor":
    if not isinstance(t, Tensor):
        t = Tensor(t, device=device)
    return t

def data_gate(fn):
    @wraps(fn)
    def wrapper(self: "Tensor", other: Data | "Tensor", *args, **kwargs):
        other = data_cast(other, self.device)
        return fn(self, other, *args, **kwargs)
    return wrapper

def device_gate(fn):
    @wraps(fn)
    def wrapper(self: "Tensor", other: "Tensor", *args, **kwargs):
        if self.device != other.device:
            raise ValueError(
                f"Tensors on different devices: {self.device} vs {other.device}"
            )
        return fn(self, other, *args, **kwargs)
    return wrapper

def input_gate(fn):
    return data_gate(device_gate(fn))

def from_op(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        props = fn(*args, **kwargs)
        return Tensor.from_props(props)
    return wrapper

def op_gate(fn):
    return input_gate(from_op(fn))


@final
class Tensor(TensorLike):
    # Class attributes to quickly reference dtype enums
    float64 = DType.float64
    float32 = DType.float32
    int64 = DType.int64
    int32 = DType.int32
    int16 = DType.int16
    int8 = DType.int8

    def __init__(
        self,
        data: Data,
        requires_grad: bool = False,
        dependencies: DependenciesList = None,
        device: Device | str = Device.CPU,
        dtype: DType = None,
    ) -> None:
        self.backend = device
        self.dtype = dtype or self.float32
        self.requires_grad = requires_grad

        self.data = self.build_ndarray(data, self.dtype, self.device)

        if self.requires_grad:
            self.grad = self.backend.zeros_like(
                self.data,
                dtype=get_dtype(self.device, self.dtype),
            )
        else:
            self.grad: Vector = None

        # Ensure proper initialization of base classes 
        self.dependencies: DependenciesList = dependencies or []


        if self.requires_grad:
            self.zero_grad()

    @classmethod
    def from_props(cls, pr: TProps | TensorLike) -> "Tensor":
        return cls(*pr.props())

    # ----------------------------
    # Core Fields
    # ----------------------------

    @property
    def dtype(self) -> DType:
        return self._dtype

    @dtype.setter
    def dtype(self, dt: DType):
        self._dtype = dt

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def props(self) -> tuple:
        return (
            self.data,
            self.requires_grad,
            self.dependencies,
            self.device,
            self.dtype
        )

    @property
    def backend(self) -> Backend:
        return self._backend

    @backend.setter
    def backend(self, device: Device | str):
        self.device = device_cast(device)
        self._backend = get_backend(self.device)

    def item(self) -> Scalar:
        r"""
        Returns the Python scalar value from a tensor with one element.
        Raises:
            ValueError: If the tensor has more than one element.
        """
        if self.data.size != 1:
            raise ValueError(
                f"Cannot convert tensor with shape {self.shape} to a scalar."
            )
        return self.data.item()

    @staticmethod
    def build_ndarray(
        data: Data,
        dtype: DType = DType.float32,
        device: Device = Device.CPU
    ) -> Vector:
        dtype_ = get_dtype(device, dtype)

        if isinstance(data, Tensor):
            return data.data  # Ensure correct return value
        
        backend = get_backend(device)
        
        # If it's a native vector (e.g., np.ndarray or cp.ndarray), pass through safely
        if isinstance(data, backend.ndarray):
            if data.dtype != dtype_:
                return data.astype(dtype_)
            return data

        # Fallback: safely convert
        return backend.array(data, dtype=dtype_)

    def to(self, device: Device | str, dtype: DType = None) -> "Tensor":
        new_device = device_cast(device)
        new_dtype = dtype or self.dtype
        new_dtype_impl = get_dtype(new_device, new_dtype)

        # Set the backend for the new device
        self.backend = new_device

        if new_device == self.device and new_dtype == self.dtype:
            return self

        if new_device != self.device:
            # Explicitly convert CuPy → NumPy using `.get()`
            if self.device == Device.CUDA and new_device == Device.CPU:
                new_data = self.data.get().astype(new_dtype_impl)
            # NumPy → CuPy should be safe via `cupy.array(...)`
            elif self.device == Device.CPU and new_device == Device.CUDA:
                new_data = self.backend.array(self.data, dtype=new_dtype_impl)
            else:
                raise RuntimeError(
                    f"Unsupported device transfer: {self.device} -> {new_device}"
                )
        else:
            # Only dtype conversion
            new_data = self.data.astype(new_dtype_impl)

        return Tensor(
            new_data,
            requires_grad=self.requires_grad,
            dependencies=self.dependencies,
            device=new_device,
            dtype=new_dtype
        )

    def zero_grad(self) -> None:
        r"""
        Zero the gradients of all parameters
        """

        if self.requires_grad:
            if self.grad is None:
                self.grad = self.backend.zeros_like(
                    self.data,
                    dtype=get_dtype(self.device, self.dtype),
                )
            else:
                self.grad.fill(0.0)

    def release_grad(self) -> None:
        r"""
        Release gradient memory
        """
        self.grad = None

    def backward(self, grad: Vector | None = None) -> None:
        if not self.requires_grad:
            raise ValueError("Backward was called on a non-required-grad tensor!")

        if grad is None:
            if self.shape == ():
                grad = self.build_ndarray(1.0, self.dtype, self.device)
            else:
                raise ValueError(
                    f"Grad must be provided for non-scalar tensors."
                    f"Tensor shape: {self.shape}"
                )

        # Device check
        if not isinstance(grad, type(self.data)):
            raise ValueError(
                f"Grad device does not match tensor device: {self.device}. "
                f"Grad must be of same backend type. "
                f"Expected {type(self.data)}, got {type(grad)}"
            )

        # Ensure `grad` has the correct shape
        if grad.shape != self.shape:
            raise ValueError(
                f"Grad shape {grad.shape} does not match tensor shape {self.shape}"
            )

        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.backend.add(self.grad, grad)

        for dependency in self.dependencies:
            backward_grad = dependency.grad_fn(grad)
            dependency.value.backward(backward_grad)

    def clip_grad(self, clip_value: float = 1.0) -> None:
        r"""
        Clips the gradients of the tensor to a specified value.
        """
        if self.requires_grad and self.grad is not None:
            self.grad = self.backend.clip(
                self.grad,
                -clip_value,
                clip_value
            )

    def clip_grad_norm(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        eps: float = 1e-6
    ) -> None:
        r"""
        Clips the gradients of the tensor to a specified norm.
        """

        if self.requires_grad and self.grad is not None:
            grad_norm = self.backend.linalg.norm(self.grad, norm_type)
            if grad_norm > max_norm:
                self.grad = self.grad * (max_norm / (grad_norm + eps))

    def save(self, file_path: str | Path) -> None:
        r"""
        Saves the tensor to a file.
        """
        state = {
            'data': self.data,
            'requires_grad': self.requires_grad,
            'grad': self.grad,
            'device': self.device.value,
            'dtype': self.dtype.value,
            # Leaf contains grad_fn, which is often non-picklable
            # (lambdas, function pointers)
            # Pickling will crash for non-trivial graphs
            # Skip dependencies when saving/loading.
            # 'dependencies': self.dependencies,
        }

        with open(file_path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, file_path: str | Path) -> "Tensor":
        r"""
        Loads a tensor from a file.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"No file found at {file_path}")

        try:
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Invalid tensor file format at {file_path}") from e

        if not isinstance(state, dict) or 'data' not in state:
            raise ValueError(f"Invalid tensor file format at {file_path}")

        device = Device(state['device'])
        dtype = DType(state['dtype'])

        tensor = cls(
            data=state['data'],
            requires_grad=state['requires_grad'],
            device=device,
            dtype=dtype,
        )

        tensor.grad = state['grad']
        # Skip dependencies when saving/loading.
        # tensor.dependencies = state['dependencies']

        return tensor

    def detach(self) -> "Tensor":
        return Tensor(
            self.data,
            device=self.device,
            requires_grad=False,
            dtype=self.dtype,
        )

    def clone(self) -> "Tensor":
        r"""
        Returns a copy of the tensor with the same data and requires_grad flag.
        """
        return Tensor(
            self.data.copy(),
            device=self.device,
            requires_grad=self.requires_grad,
            dtype=self.dtype,
            dependencies=(
                self.dependencies.copy()
                if self.dependencies
                else None
            ),
        )

    def copy(self) -> "Tensor":
        r"""
        Alias for clone().
        """
        return self.clone()

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data}, requires_grad={self.requires_grad}, "
            f"shape={self.shape}, dtype={self.dtype}, device={self.device})"
        )

    @staticmethod
    def randn(
        dims: Dims = (),
        requires_grad = False,
        device: Device = Device.CPU,
    ) -> "Tensor":
        backend = get_backend(device)

        if isinstance(dims, int):
            dims = (dims,)

        data = backend.random_randn(dims)
        return Tensor(
            data,
            requires_grad=requires_grad,
            device=device,
        )

    @staticmethod
    def uniform(
        low: float,
        high: float,
        dims: Dims = (),
        requires_grad = False,
        device: Device = Device.CPU,
    ) -> "Tensor":
        backend = get_backend(device)

        if type(dims) is int:
            dims = (dims,)

        data = backend.random_uniform(low, high, dims)
        return Tensor(
            data,
            requires_grad=requires_grad,
            device=device,
        )

    @from_op
    def view(self, shape: Shape) -> "Tensor":
        return Ops.base_ops.view(self, shape)

    def reshape(self, shape: Shape) -> "Tensor":
        return self.view(shape)

    @from_op
    def broadcast_to(self, shape: Shape) -> "Tensor":
        return Ops.base_ops.broadcast_to(self, shape)

    @from_op
    def transpose(self, axis: Shape = None) -> "Tensor":
        return Ops.base_ops.transpose(self, axis)

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    @from_op
    def squeeze(self, dim: int | tuple[int] = 0) -> "Tensor":
        return Ops.base_ops.squeeze(self, dim)

    @from_op
    def unsqueeze(self, dim: tuple[int] = 0) -> "Tensor":
        return Ops.base_ops.unsqueeze(self, dim)

    ###########################################################################
    ############################## Elementwise Ops ############################
    ###########################################################################

    @staticmethod
    @from_op
    def where(condition: "Tensor", a: "Tensor", b: "Tensor") -> "Tensor":
        return Ops.elementwise_ops.where(condition, a, b)

    @staticmethod
    @op_gate
    def maximum(a: "Tensor", b: "Tensor") -> "Tensor":
        return Ops.elementwise_ops.maximum(a, b)

    @staticmethod
    @op_gate
    def minimum(a: "Tensor", b: "Tensor") -> "Tensor":
        return Ops.elementwise_ops.minimum(a, b)

    @from_op
    def abs(self) -> "Tensor":
        return Ops.elementwise_ops.abs(self)

    def threshold(self, threshold: float, value: float) -> "Tensor":
        return Tensor.where(self > threshold, self, Tensor(value))

    @input_gate
    def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
        return Tensor.where(mask, Tensor(value), self)

    def sign(self) -> "Tensor":
        return Tensor.where(
            self > 0, Tensor(1),
            Tensor.where(self < 0, Tensor(-1), Tensor(0))
        )

    def clip(
        self, min_value: float = None, max_value: float = None
    ) -> "Tensor":
        return Tensor.where(
            self < min_value, Tensor(min_value),
            Tensor.where(self > max_value, Tensor(max_value), self)
        )

    ###########################################################################
    ############################## Operator Overload ##########################
    ###########################################################################

    @from_op
    def __getitem__(self, index: Index) -> "Tensor":
        return Ops.overload_ops.get_item(self, index)

    ### Comparison Operators ###
    @data_gate
    def __lt__(self, other: Data) -> "Tensor":
        return Tensor(self.data < other.data)

    @data_gate
    def __gt__(self, other: Data) -> "Tensor":
        return Tensor(self.data > other.data)

    @data_gate
    def __eq__(self, other: Data) -> "Tensor":
        return Tensor(self.data == other.data)

    @data_gate
    def __le__(self, other: Data) -> "Tensor":
        return Tensor(self.data <= other.data)

    @data_gate
    def __ge__(self, other: Data) -> "Tensor":
        return Tensor(self.data >= other.data)

    @data_gate
    def __ne__(self, other: Data) -> "Tensor":
        return Tensor(self.data != other.data)

    ### Math Operators ###
    @op_gate
    def __add__(self, other: Data) -> "Tensor":
        return Ops.overload_ops.add(self, other)

    @op_gate
    def __radd__(self, other: Data) -> "Tensor":
        return Ops.overload_ops.add(other, self)

    def __iadd__(self, other: Data) -> "Tensor":
        other = Tensor.build_ndarray(other, device=self.device)
        self.data = self.backend.add(self.data, other)
        return self

    @from_op
    def __neg__(self) -> "Tensor":
        return Ops.overload_ops.neg(self)

    @input_gate
    def __sub__(self, other: Data) -> "Tensor":
        return self + (-other)

    @input_gate
    def __rsub__(self, other: Data) -> "Tensor":
        return other + (-self)

    def __isub__(self, other: Data) -> "Tensor":
        r"""
        In-place subtraction self: -= other
        There is no gradient function for in-place operations!
        """
        other = -Tensor.build_ndarray(other, device=self.device)
        self.data = self.backend.add(self.data, other)
        return self

    @op_gate
    def __mul__(self, other: Data) -> "Tensor":
        return Ops.overload_ops.mul(self, other)

    @op_gate
    def __rmul__(self, other: Data) -> "Tensor":
        return Ops.overload_ops.mul(other, self)

    def __imul__(self, other: Data) -> "Tensor":
        r"""
        In-place multiplication self: *= other
        There is no gradient function for in-place operations!
        """
        other = Tensor.build_ndarray(other, device=self.device)
        self.data = self.backend.multiply(self.data, other)
        return self

    @op_gate
    def __matmul__(self, other: Data) -> "Tensor":
        return Ops.overload_ops.matmul(self, other)

    @op_gate
    def __rmatmul__(self, other: Data) -> "Tensor":
        return Ops.overload_ops.matmul(other, self)

    def __pow__(self, pow: int) -> "Tensor":
        return self.pow(pow)

    @input_gate
    def __truediv__(self, other: Data) -> "Tensor":
        return self * (other**-1)

    @input_gate
    def __rtruediv__(self, other: Data) -> "Tensor":
        return other * (self**-1)

    def __itruediv__(self, other: Data) -> "Tensor":
        r"""
        In-place division self: /= other
        There is no gradient function for in-place operations!
        """
        other = Tensor.build_ndarray(other, device=self.device)
        self.data = self.backend.true_divide(self.data, other)
        return self

    ###########################################################################
    ################################# Math Ops ################################
    ###########################################################################

    @from_op
    def sum(self, axis: int = None, keepdims: bool = False) -> "Tensor":
        return Ops.reduce_ops.sum(self, axis, keepdims)

    @from_op
    def mean(self) -> "Tensor":
        return Ops.reduce_ops.mean(self)

    @from_op
    def max(self, axis: int = None, keepdims: bool = False) -> "Tensor":
        return Ops.reduce_ops.max(self, axis, keepdims)

    @from_op
    def log(self) -> "Tensor":
        return Ops.math_ops.log(self)

    @from_op
    def exp(self) -> "Tensor":
        return Ops.math_ops.exp(self)

    @from_op
    def pow(self, pow: int) -> "Tensor":
        return Ops.math_ops.pow(self, pow)

    @from_op
    def sqrt(self) -> "Tensor":
        return Ops.math_ops.pow(self, 0.5)

    @from_op
    def tanh(self) -> "Tensor":
        return Ops.math_ops.tanh(self)
