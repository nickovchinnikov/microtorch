from src.tensor.backend import get_backend


class Ops:
    r"""
    Base class for all operations.
    """

    def __init__(self, device: str):
        r"""
        Initializes the backend for the operation.
        
        Args:
            device (str): The device to use for the operation.
        """
        self.backend = get_backend(device)
