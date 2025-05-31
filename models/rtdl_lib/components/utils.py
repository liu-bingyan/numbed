import torch
import torch.nn as nn
from typing import Any, List, Type, Union

ModuleType = Union[str, Type[nn.Module]]

def _all_or_none(iterable: List[Any]) -> bool:
    """Return True if all elements are None or if all elements are not None."""
    return all(x is None for x in iterable) or all(x is not None for x in iterable)

def _is_glu_activation(activation: ModuleType) -> bool:
    """Check if the activation is a GLU variant."""
    return activation in ['GEGLU', 'ReGLU']

def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    """Create a PyTorch module based on the module type."""
    if isinstance(module_type, str):
        if module_type == 'Identity':
            return nn.Identity()
        elif module_type == 'ReLU':
            return nn.ReLU()
        elif module_type == 'GELU':
            return nn.GELU()
        elif module_type == 'Sigmoid':
            return nn.Sigmoid()
        elif module_type == 'Tanh':
            return nn.Tanh()
        else:
            raise ValueError(f'Unknown module type: {module_type}')
    else:
        return module_type(*args)

class _TokenInitialization:
    """Helper class for token initialization."""

    @staticmethod
    def from_str(initialization: str) -> '_TokenInitialization':
        if initialization == 'uniform':
            return _UniformInitialization()
        elif initialization == 'normal':
            return _NormalInitialization()
        else:
            raise ValueError(f'Unknown initialization: {initialization}')

    def apply(self, parameter: nn.Parameter, d: int) -> None:
        raise NotImplementedError

class _UniformInitialization(_TokenInitialization):
    """Uniform initialization for tokens."""

    def apply(self, parameter: nn.Parameter, d: int) -> None:
        nn.init.uniform_(parameter, -d ** -0.5, d ** -0.5)

class _NormalInitialization(_TokenInitialization):
    """Normal initialization for tokens."""

    def apply(self, parameter: nn.Parameter, d: int) -> None:
        nn.init.normal_(parameter, 0, d ** -0.5) 