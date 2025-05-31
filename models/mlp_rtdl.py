import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basemodel_torch import BaseModelTorch

'''
    Custom implementation for the standard multi-layer perceptron
'''


class MLP(BaseModelTorch):

    def __init__(self, params, args):
        super().__init__(params, args)
        
        self.model = MLP_Model(
            d_in=self.adjusted_input_dim, 
            d_layers=self.params["d_layers"], 
            d_out=self.args.num_classes,
            dropouts=self.params.get("dropout", 0.0),
            task_type=self.args.objective
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
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 100),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.001)
        }
        return params

class MLP_Model(nn.Module):
    class Block(nn.Module):
        def __init__(self, d_in, d_out, bias, dropout, task_type):
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = nn.ReLU()
            # Only apply dropout for classification tasks
            self.use_dropout = dropout > 0 and (task_type != 'regression')
            if self.use_dropout:
                self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.activation(self.linear(x))
            if self.use_dropout:
                x = self.dropout(x)
            return x

    def __init__(self, d_in, d_layers, dropouts, d_out, task_type):
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        
        self.blocks = nn.Sequential(
            *[MLP_Model.Block(
                d_in=d_layers[i - 1] if i else d_in,
                d_out=d,
                bias=True,
                dropout=dropout,
                task_type=task_type,
            ) for i, (d, dropout) in enumerate(zip(d_layers, dropouts))]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x):
        x = self.blocks(x)
        x = self.head(x)
        return x



