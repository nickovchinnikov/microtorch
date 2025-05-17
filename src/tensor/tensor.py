import pickle
from functools import wraps
from pathlib import Path
from typing import Optional, Tuple, TypeGuard, Union, final

from .device import Device, _device
from .backend import get_backend, Backend, DType, get_dtype, Vector, Scalar
from .ops import BaseOps, Elementwise, MathOps, OverloadOps, Reduce
from .types import Data, DependenciesList, Dims, Index, Shape, TensorLike, TProps


def is_tensorlike(obj: object) -> TypeGuard[TensorLike]:
    return isinstance(obj, Tensor)

def data_cast(t: Union[Data, "Tensor"], device: Device) -> "Tensor":
    if not isinstance(t, Tensor):
        t = Tensor(t, device=device)
    return t

def data_gate(fn):
    @wraps(fn)
    def wrapper(self: "Tensor", other: Union[Data, "Tensor"], *args, **kwargs):
        other = data_cast(other, self.device)
        return fn(self, other, *args, **kwargs)
    return wrapper

def device_gate(fn):
    @wraps(fn)
    def wrapper(self: "Tensor", other: "Tensor", *args, **kwargs):
        if self.device != other.device:
            raise ValueError(f"Tensors on different devices: {self.device} vs {other.device}")
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
    float64 = DType.FLOAT64
    float32 = DType.FLOAT32
    int64 = DType.INT64
    int32 = DType.INT32
    int16 = DType.INT16
    int8 = DType.INT8

    def __init__(
        self,
        data: Data,
        requires_grad: bool = False,
        dependencies: Optional[DependenciesList] = None,
        device: Union[Device, str] = Device.CPU,
        dtype: Optional[DType] = None,
    ) -> None:
        self.device = _device(device)
        self._backend = get_backend(self.device)
        self.dtype = dtype or self.float32

        # Ensure proper initialization of base classes 
        self.dependencies: DependenciesList = dependencies or []

        self._data = self.build_ndarray(data, self.dtype, self.device)

        self.requires_grad = requires_grad

        if self.requires_grad:
            self.grad = self.backend.zeros_like(
                self.data,
                dtype=get_dtype(self.device, self.dtype),
            )
        else:
            self.grad: Vector = None

        if self.requires_grad:
            self.zero_grad()

    @classmethod
    def from_props(cls, prps: Union[TProps, TensorLike]) -> "Tensor":
        return cls(*prps.props())

    # ----------------------------
    # Core Fields
    # ----------------------------

    @property
    def data(self) -> Vector:
        r"""Return the data of the tensor"""
        return self._data

    @data.setter
    def data(self, new_data: Data) -> None:
        r"""Set the data of the tensor"""
        self._data = self.build_ndarray(new_data, self.dtype, self.device)
        self.zero_grad()

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, rg: bool):
        self._requires_grad = rg

    @property
    def dependencies(self) -> DependenciesList:
        return self._dependencies

    @dependencies.setter
    def dependencies(self, deps: DependenciesList):
        self._dependencies = deps

    @property
    def device(self) -> Device:
        return self._device

    @device.setter
    def device(self, dev: Device):
        self._device = dev

    @property
    def dtype(self) -> DType:
        return self._dtype

    @dtype.setter
    def dtype(self, dt: DType):
        self._dtype = dt

    @property
    def grad(self) -> Optional[Vector]:
        return self._grad

    @grad.setter
    def grad(self, gr: Vector):
        self._grad = gr

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim

    def props(self) -> Tuple:
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
    def backend(self, device: Device):
        self.device = device
        self.backend = get_backend(self.device)

    def item(self) -> Scalar:
        r"""
        Returns the Python scalar value from a tensor with one element.
        Raises:
            ValueError: If the tensor has more than one element.
        """
        if self.data.size != 1:
            raise ValueError(f"Cannot convert tensor with shape {self.shape} to a scalar.")
        return self.data.item()

    @staticmethod
    def build_ndarray(
        data: Data,
        dtype: DType = DType.FLOAT32,
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

    def to(self, device: Union[Device, str], dtype: DType = None) -> "Tensor":
        new_device = _device(device)
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
                raise RuntimeError(f"Unsupported device transfer: {self.device} -> {new_device}")
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
                    self._data,
                    dtype=get_dtype(self.device, self.dtype),
                )
            else:
                self.grad.fill(0.0)

    def release_grad(self) -> None:
        r"""
        Release gradient memory
        """
        self.grad = None

    def backward(self, grad: Optional[Vector] = None) -> None:
        if not self.requires_grad:
            raise ValueError("Backward was called on a non-required-grad tensor!")

        if grad is None:
            if self.shape == ():
                grad = self.build_ndarray(1.0, self.dtype, self.device)
            else:
                raise ValueError(f"Grad must be provided for non-scalar tensors. Tensor shape: {self.shape}")

        # Device check
        if not isinstance(grad, type(self.data)):
            raise ValueError((f"Grad device does not match tensor device: {self.device}. "
                              "Grad must be of same backend type. "
                              f"Expected {type(self.data)}, got {type(grad)}"))

        # Ensure `grad` has the correct shape
        if grad.shape != self.shape:
            raise ValueError(f"Grad shape {grad.shape} does not match tensor shape {self.shape}")

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

    def save(self, file_path: Union[str, Path]) -> None:
        r"""
        Saves the tensor to a file.
        """
        state = {
            'data': self.data,
            'requires_grad': self.requires_grad,
            'grad': self.grad,
            'device': self.device.value,
            'dtype': self.dtype.value,
            # Leaf contains grad_fn, which is often non-picklable (lambdas, function pointers)
            # Pickling will crash for non-trivial graphs
            # Skip dependencies when saving/loading.
            # 'dependencies': self.dependencies,
        }

        with open(file_path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "Tensor":
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
            f"Tensor(data={self.data}, requires_grad={self.requires_grad}, shape={self.shape}, dtype={self.dtype}, device={self.device})"
        )

    @staticmethod
    def randn(
        dims: Dims = (),
        requires_grad = False,
        device: Device = Device.CPU,
    ) -> "Tensor":
        backend = get_backend(device)

        if type(dims) is int:
            dims = (dims,)

        data = backend.random_randn(*dims)
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
        return BaseOps.view(self, shape)

    def reshape(self, shape: Shape) -> "Tensor":
        return self.view(shape)

    @from_op
    def broadcast_to(self, shape: Shape) -> "Tensor":
        return BaseOps.broadcast_to(self, shape)

    @from_op
    def transpose(self, axis: Shape = None) -> "Tensor":
        return BaseOps.transpose(self, axis)

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    @from_op
    def squeeze(self, dim: int | Tuple[int] = 0) -> "Tensor":
        return BaseOps.squeeze(self, dim)

    @from_op
    def unsqueeze(self, dim: Tuple[int] = 0) -> "Tensor":
        return BaseOps.unsqueeze(self, dim)

    ###########################################################################
    ############################## Elementwise Ops ############################
    ###########################################################################

    @staticmethod
    @from_op
    def where(condition: "Tensor", a: "Tensor", b: "Tensor") -> "Tensor":
        return Elementwise.where(condition, a, b)

    @staticmethod
    @op_gate
    def maximum(a: "Tensor", b: "Tensor") -> "Tensor":
        return Elementwise.maximum(a, b)

    @staticmethod
    @op_gate
    def minimum(a: "Tensor", b: "Tensor") -> "Tensor":
        return Elementwise.minimum(a, b)

    @from_op
    def abs(self) -> "Tensor":
        return Elementwise.abs(self)

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

    def clip(self, min_value: Optional[float] = None, max_value: Optional[float] = None) -> "Tensor":
        return Tensor.where(
            self < min_value, Tensor(min_value),
            Tensor.where(self > max_value, Tensor(max_value), self)
        )

    ###########################################################################
    ############################## Operator Overload ##########################
    ###########################################################################

    @from_op
    def __getitem__(self, index: Index) -> "Tensor":
        return OverloadOps.get_item(self, index)

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
        return OverloadOps.add(self, other)

    @op_gate
    def __radd__(self, other: Data) -> "Tensor":
        return OverloadOps.add(other, self)

    def __iadd__(self, other: Data) -> "Tensor":
        other = Tensor.build_ndarray(other, device=self.device)
        self.backend.add(self.data, other, out=self.data)
        return self

    @from_op
    def __neg__(self) -> "Tensor":
        return OverloadOps.neg(self)

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
        self.backend.add(self.data, other, out=self.data)
        return self

    @op_gate
    def __mul__(self, other: Data) -> "Tensor":
        return OverloadOps.mul(self, other)

    @op_gate
    def __rmul__(self, other: Data) -> "Tensor":
        return OverloadOps.mul(other, self)

    def __imul__(self, other: Data) -> "Tensor":
        r"""
        In-place multiplication self: *= other
        There is no gradient function for in-place operations!
        """
        other = Tensor.build_ndarray(other, device=self.device)
        self.backend.multiply(self.data, other, out=self.data)
        return self

    @op_gate
    def __matmul__(self, other: Data) -> "Tensor":
        return OverloadOps.matmul(self, other)

    @op_gate
    def __rmatmul__(self, other: Data) -> "Tensor":
        return OverloadOps.matmul(other, self)

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
        self.backend.true_divide(self.data, other, out=self.data)
        return self

    ###########################################################################
    ################################# Math Ops ################################
    ###########################################################################

    @from_op
    def sum(self, axis: int = None, keepdims: bool = False) -> "Tensor":
        return Reduce.sum(self, axis, keepdims)

    @from_op
    def mean(self) -> "Tensor":
        return Reduce.mean(self)

    @from_op
    def max(self, axis: int = None, keepdims: bool = False) -> "Tensor":
        return Reduce.max(self, axis, keepdims)

    @from_op
    def log(self) -> "Tensor":
        return MathOps.log(self)

    @from_op
    def exp(self) -> "Tensor":
        return MathOps.exp(self)

    @from_op
    def pow(self, pow: int) -> "Tensor":
        return MathOps.pow(self, pow)

    @from_op
    def sqrt(self) -> "Tensor":
        return MathOps.pow(self, 0.5)

    @from_op
    def tanh(self) -> "Tensor":
        return MathOps.tanh(self)
