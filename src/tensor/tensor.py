import pickle
from functools import wraps
from pathlib import Path
from typing import Optional, Tuple, Union

from .device import Device, DType, Vector, _device, _tensor, get_dtype
from .ops import BaseOps, Elementwise, MathOps, OverloadOps, Reduce
from .types import Data, DependenciesList, TensorLike, TProps


def data_cast(t: Union[Data, "Tensor"]) -> "Tensor":
    if not isinstance(t, Tensor):
        t = Tensor(t)
    return t

def data_gate(fn):
    @wraps(fn)
    def wrapper(self: Tensor, other: Union[Data, Tensor], *args, **kwargs):
        other = data_cast(other)
        return fn(self, other, *args, **kwargs)
    return wrapper

def device_gate(fn):
    @wraps(fn)
    def wrapper(self: Tensor, other: Tensor, *args, **kwargs):
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
        dtype: Optional[TensorLike] = None,
    ) -> None:
        self.device = _device(device)

        # Ensure proper initialization of base classes 
        self.dependencies: DependenciesList = dependencies or []
        self.dtype = dtype or self.float32

        self._data = self.build_ndarray(data, self.dtype, self.device)

        self.requires_grad = requires_grad

        if self.requires_grad:
            self.grad = _tensor(self.device).zeros_like(
                self._data,
                dtype=get_dtype(self.device, self.dtype),
            )
        else:
            self.grad: Vector = None

        if self.requires_grad:
            self.zero_grad()

    @classmethod
    def from_props(cls, prps: TProps) -> "Tensor":
        return cls(*prps.props())

    @property
    def data(self) -> Vector:
        r"""Return the data of the tensor"""
        return self._data

    @data.setter
    def data(self, new_data: Data) -> None:
        r"""Set the data of the tensor"""
        self._data = self.build_ndarray(new_data, self.dtype, self.device)
        self.zero_grad()

    @staticmethod
    def build_ndarray(
        data: Data,
        dtype: DType = DType.FLOAT32,
        device: Device = Device.CPU
    ) -> Vector:
        dtype_ = get_dtype(device, dtype)

        if isinstance(data, Tensor):
            return data.data  # Ensure correct return value

        return _tensor(device).array(data, dtype=dtype_)

    def to(self, device: Union[Device, str], dtype: DType = None) -> "Tensor":
        new_device = _device(device)
        new_dtype = dtype or self.dtype
        new_dtype_impl = get_dtype(new_device, new_dtype)

        # If no change is needed, return self
        if new_device == self.device and new_dtype == self.dtype:
            return self

        # Handle device transfer
        if new_device != self.device:
            new_data = _tensor(new_device).array(
                self.data, dtype=new_dtype_impl
            )
        else:
            new_data = self.data

        # Handle dtype conversion
        if new_dtype != self.dtype:
            new_data = new_data.astype(new_dtype_impl)

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

        if self.grad is None:
            self.grad = _tensor(self.device).zeros_like(
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

        # Ensure `grad` has the correct shape
        if grad.shape != self.shape:
            raise ValueError(f"Grad shape {grad.shape} does not match tensor shape {self.shape}")

        if grad.device != self.device:
            raise ValueError(f"Grad device {grad.device} does not match tensor device {self.device}")

        if self.grad is None:
            self.grad = grad
        else:
            _tensor(self.device).add(self.grad, grad, out=self.grad)

        for dependency in self.dependencies:
            backward_grad = dependency.grad_fn(grad)
            dependency.value.backward(backward_grad)

    def clip_grad(self, clip_value: float = 1.0) -> None:
        r"""
        Clips the gradients of the tensor to a specified value.
        """

        if self.requires_grad and self.grad is not None:
            self.grad = _tensor(self.device).clip(
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
            grad_norm = _tensor(self.device).linalg.norm(self.grad, norm_type)
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
        r"""
        Returns a new tensor, detached from the current computational graph.
        """
        return Tensor(
            self.data.copy(),
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
        dims: Tuple[int] | int = (),
        requires_grad=False,
        device: Device = Device.CPU,
    ) -> "Tensor":
        if type(dims) is int:
            dims = (dims,)
        return Tensor(
            _tensor(device).random.randn(*dims),
            requires_grad=requires_grad,
            device=device,
        )

    def view(self, shape: Tuple[int, ...]) -> "Tensor":
        return Tensor.from_props(BaseOps.view(self, shape))

    def transpose(self, axis: Tuple[int, ...] = None) -> "Tensor":
        return Tensor.from_props(BaseOps.transpose(self, axis))

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def squeeze(self, dim: int | Tuple[int] = 0) -> "Tensor":
        return Tensor.from_props(BaseOps.squeeze(self, dim))

    def unsqueeze(self, dim: Tuple[int] = 0) -> "Tensor":
        return Tensor.from_props(BaseOps.unsqueeze(self, dim))

    ###########################################################################
    ############################## Elementwise Ops ############################
    ###########################################################################

    @staticmethod
    def where(condition: "Tensor", a: "Tensor", b: "Tensor") -> "Tensor":
        return Tensor.from_props(Elementwise.where(condition, a, b))

    @staticmethod
    @op_gate
    def maximum(a: "Tensor", b: "Tensor") -> "Tensor":
        return Elementwise.maximum(a, b)

    @staticmethod
    @op_gate
    def minimum(a: "Tensor", b: "Tensor") -> "Tensor":
        return Elementwise.minimum(a, b)

    def abs(self) -> "Tensor":
        return Tensor.from_props(Elementwise.abs(self))

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

    @data_gate
    def __getitem__(self, index: Union[int, slice, Tuple[Union[int, slice]], "Tensor", Vector]) -> "Tensor":
        return Tensor.from_props(
            OverloadOps.get_item(self, index.data)
        )

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
        other = Tensor.build_ndarray(other)
        _tensor(self.device).add(self.data, other, out=self.data)
        return self

    def __neg__(self) -> "Tensor":
        return Tensor.from_props(OverloadOps.neg(self))

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
        other = -Tensor.build_ndarray(other)
        _tensor(self.device).add(self.data, other, out=self.data)
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
        other = Tensor.build_ndarray(other)
        _tensor(self.device).multiply(self.data, other, out=self.data)
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
        other = Tensor.build_ndarray(other)
        _tensor(self.device).true_divide(self.data, other, out=self.data)
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
    def tanh(self) -> "Tensor":
        return MathOps.tanh(self)
