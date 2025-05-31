import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel_torch import BaseModelTorch

'''
    Faithful implementation of ResNet from modules.py
'''

class ResNet(BaseModelTorch):
    def __init__(self, params, args):
        super().__init__(params, args)
        
        self.model = ResNet_Model(
            d_in=self.adjusted_input_dim,
            n_blocks=self.params.get("n_blocks", 3),
            d_main=self.params.get("d_main", 64),
            d_hidden=self.params.get("d_hidden", 128),
            dropout_first=self.params.get("dropout_first", 0.1),
            dropout_second=self.params.get("dropout_second", 0.0),
            d_out=self.args.num_classes
        )

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        return super().fit(X, y, X_val, y_val)

    def predict_helper(self, X):
        X = np.array(X, dtype=np.float32)
        return super().predict_helper(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_blocks": trial.suggest_int("n_blocks", 2, 5),
            "d_main": trial.suggest_int("d_main", 32, 256),
            "d_hidden": trial.suggest_int("d_hidden", 64, 512),
            "dropout_first": trial.suggest_float("dropout_first", 0.0, 0.5),
            "dropout_second": trial.suggest_float("dropout_second", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.001)
        }
        return params


class ResidualBlock(nn.Module):
    """The main building block of ResNet, following the implementation in modules.py."""
    
    def __init__(
        self,
        *,
        d_main: int,
        d_hidden: int,
        bias_first: bool = True,
        bias_second: bool = True,
        dropout_first: float = 0.0,
        dropout_second: float = 0.0,
        normalization: str = 'BatchNorm1d',
        activation: str = 'ReLU',
        skip_connection: bool = True,
    ) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm1d(d_main)
        self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
        self.activation = nn.ReLU()
        self.dropout_first = nn.Dropout(dropout_first)
        self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
        self.dropout_second = nn.Dropout(dropout_second)
        self.skip_connection = skip_connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x
        x = self.normalization(x)
        x = self.linear_first(x)
        x = self.activation(x)
        x = self.dropout_first(x)
        x = self.linear_second(x)
        x = self.dropout_second(x)
        if self.skip_connection:
            x = x_input + x
        return x


class Head(nn.Module):
    """The final module of ResNet, following the implementation in modules.py."""
    
    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        bias: bool = True,
        normalization: str = 'BatchNorm1d',
        activation: str = 'ReLU',
    ) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm1d(d_in)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(d_in, d_out, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization(x)
        x = self.activation(x)
        x = self.linear(x)
        return x


class ResNet_Model(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        d_out: int,
    ) -> None:
        super().__init__()
        
        # First layer
        self.first_layer = nn.Linear(d_in, d_main)
        
        # Residual blocks
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization='BatchNorm1d',
                    activation='ReLU',
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        
        # Head module
        self.head = Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization='BatchNorm1d',
            activation='ReLU',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x