import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basemodel_torch import BaseModelTorch
from models.basemodule_embedding import Tokenizer
import math
from models.rtdl_lib.modules import Transformer as RTDLTransformer, CLSToken, FeatureTokenizer

class RTDLTokenizer(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, d_token):
        super().__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        if not cat_cardinalities:
            self.cat_cardinalities = None
        self.tokenzier = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token)
    def forward(self, x):
        x_num = x[:, :self.n_num_features]
        x_cat = x[:, self.n_num_features:]
        if self.cat_cardinalities is None:
            x_cat = None
        tokenized = self.tokenzier(x_num,x_cat)
        return tokenized
        
class TransformerModel(nn.Module):
    def __init__(self, params, args):
        super().__init__()
        self.params = params
        self.args = args
        self.tokenizer = Tokenizer(args)
                        # RTDLTokenizer(n_num_features= len(self.args.num_idx),
                        #                     cat_cardinalities = self.args.cat_dims,
                        #                     d_token=self.params["num_bins"]
                        #                 )
        self.clsToken = CLSToken(d_token=self.params["num_bins"], initialization='uniform')
        self.transformer =  RTDLTransformer(
                                d_out = 1 if self.args.objective == 'regression' else self.args.num_classes,
                                n_blocks = self.params["n_blocks"],            
                                d_token=self.params["num_bins"],
                                residual_dropout=self.params["residual_dropout"],
                                attention_dropout=self.params["attention_dropout"],
                                ffn_dropout=self.params["ffn_dropout"],
                                ffn_d_hidden=self.params["ffn_d_hidden"],             
                                attention_n_heads = 8,
                                prenormalization = True,
                                first_prenormalization = False,
                                attention_initialization = 'kaiming',
                                attention_normalization = 'LayerNorm',
                                ffn_activation = 'ReGLU',
                                ffn_normalization = 'LayerNorm',
                                head_activation = 'ReLU',
                                head_normalization = 'LayerNorm',
                                n_tokens = None, # will be set automatically
                                kv_compression_ratio = None,
                                kv_compression_sharing = None,
                                last_layer_query_idx = None,                        
                            )
        
    def forward(self, x):
        x = self.tokenizer(x)
        x = self.clsToken(x)
        x = self.transformer(x)
        return x
    
class Transformer(BaseModelTorch):
    def __init__(self, params, args):
        super().__init__(params, args)
        # params_format:
        #    "transformer-t-lr": {
        #         "residual_dropout": 0.0,
        #         "n_blocks": 3,
        #         "attention_dropout": 0.24799761595511277,
        #         "ffn_dropout": 0.2673643110676694,
        #         "ffn_d_hidden": 942,
        #         "num_bins": 512,
        #         "learning_rate": 1.7926321761347656e-05
        #     }
        
        self.model = TransformerModel(params,args)
        
    def fit(self, X, y, X_val=None, y_val=None):
        X = np.array(X, dtype=np.float32)
        if X_val is not None:
            X_val = np.array(X_val, dtype=np.float32)
        return super().fit(X, y, X_val, y_val)

    def predict_helper(self, X):
        X = np.array(X, dtype=np.float32)
        return super().predict_helper(X)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "residual_dropout": trial.suggest_float("residual_dropout", 0.0, 0.5),
            "n_blocks": trial.suggest_int("n_blocks", 1, 4),
            "attention_dropout": trial.suggest_float("attention_dropout", 0.0, 0.5),
            "ffn_dropout": trial.suggest_float("ffn_dropout", 0.0, 0.5),
            "ffn_d_hidden": trial.suggest_int("ffn_d_hidden", 128, 1024),
            "num_bins": trial.suggest_int("num_bins", 128, 512),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "prenormalization": trial.suggest_categorical("prenormalization", [True, False]),
            "first_prenormalization": trial.suggest_categorical("first_prenormalization", [True, False]),
            "attention_initialization": trial.suggest_categorical("attention_initialization", ["kaiming", "xavier"]),
            "attention_normalization": trial.suggest_categorical("attention_normalization", ["LayerNorm", "BatchNorm1d"]),
            "ffn_activation": trial.suggest_categorical("ffn_activation", ["ReGLU", "GELU", "ReLU"]),
            "ffn_normalization": trial.suggest_categorical("ffn_normalization", ["LayerNorm", "BatchNorm1d"]),
            "head_activation": trial.suggest_categorical("head_activation", ["ReLU", "GELU", "SELU"]),
            "head_normalization": trial.suggest_categorical("head_normalization", ["LayerNorm", "BatchNorm1d"]),
        }
        return params


