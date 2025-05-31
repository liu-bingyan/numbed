import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basemodel_torch import BaseModelTorch
import math

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
        self.model = Transformer_Model(
            n_tokens = self.args.num_features,
            objective = self.args.objective,
            num_classes = self.args.num_classes,
            # required parameters
            residual_dropout=self.params["residual_dropout"],
            n_blocks=self.params["n_blocks"],
            attention_dropout=self.params["attention_dropout"],
            ffn_dropout=self.params["ffn_dropout"],
            ffn_d_hidden=self.params["ffn_d_hidden"],
            d_tokens=self.params["num_bins"],
            # optional parameters
            prenormalization=self.params.get("prenormalization", True),
            first_prenormalization=self.params.get("first_prenormalization", False),
            attention_initialization=self.params.get("attention_initialization", "kaiming"),
            attention_normalization=self.params.get("attention_normalization", "LayerNorm"),
            ffn_activation=self.params.get("ffn_activation", "ReGLU"),
            ffn_normalization=self.params.get("ffn_normalization", "LayerNorm"),
            head_activation=self.params.get("head_activation", "ReLU"),
            head_normalization=self.params.get("head_normalization", "LayerNorm"),
            kv_compression_ratio=self.params.get("kv_compression_ratio", None),
            kv_compression_sharing=self.params.get("kv_compression_sharing", None),
        )
        
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

def make_nn_module(module_type, d_model):
    if module_type == "LayerNorm":
        return nn.LayerNorm(d_model)
    elif module_type == "BatchNorm1d":
        return nn.BatchNorm1d(d_model)
    else:
        raise ValueError(f"Unknown normalization: {module_type}")
    
class ReGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.relu(a) * b
    
def init_reglu_weights(module):
    # Since ReGLU splits the input in half and applies ReLU to one half,
    # we need custom initialization for both parts
    with torch.no_grad():
        if isinstance(module, nn.Linear):
            fan_in = module.in_features
            # Calculate proper scaling for ReGLU
            # The ReLU half needs sqrt(2) gain like normal ReLU
            # The gate half needs a smaller scaling
            if module.out_features % 2 == 0:  # Make sure output can be split in half
                # Initialize first half (ReLU part) with Kaiming/He
                relu_gain = math.sqrt(2.0)
                std = relu_gain / math.sqrt(fan_in)
                nn.init.normal_(module.weight[:module.out_features//2], 0, std)
                
                # Initialize second half (gate part) with smaller variance
                gate_gain = 1.0  # Use standard normal for gate part
                gate_std = gate_gain / math.sqrt(fan_in)
                nn.init.normal_(module.weight[module.out_features//2:], 0, gate_std)
                
                # Initialize bias if it exists
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # Fallback for odd output sizes
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')


def make_activation(activation_type):
    if activation_type == "ReLU":
        return nn.ReLU()
    elif activation_type == "GELU":
        return nn.GELU()
    elif activation_type == "SELU":
        return nn.SELU()
    elif activation_type == "ReGLU":
        return ReGLU()
    else:
        raise ValueError(f"Unknown activation: {activation_type}")
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, kv_compression=None, initialization="kaiming"):
        super().__init__()
        # Check that d_model is divisible by n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        for parameter in [self.W_q.weight, self.W_k.weight, self.W_v.weight, self.W_o.weight]:
            if initialization == "kaiming":
                nn.init.kaiming_uniform_(parameter, a=2 ** 0.5)
            elif initialization == "xavier":
                nn.init.xavier_uniform_(parameter)
            else:
                raise ValueError(f"Unknown initialization: {initialization}")
        
        # KV compression
        self.kv_compression = kv_compression
        self.k_compression = None
        self.v_compression = None
        
        # Initialize compression layers if kv_compression is provided
        if kv_compression is not None:
            # Create compression layers for keys and values
            # The compression is applied to the sequence length dimension
            self.k_compression = nn.Linear(kv_compression['n_tokens'], 
                                         int(kv_compression['n_tokens'] * kv_compression['ratio']), 
                                         bias=False)
            if kv_compression['sharing'] == 'headwise':
                self.v_compression = nn.Linear(kv_compression['n_tokens'], 
                                             int(kv_compression['n_tokens'] * kv_compression['ratio']), 
                                             bias=False)
            else:
                self.v_compression = self.k_compression
        
    def forward(self, x, mask=None):
        # # Validate input dimensions
        # if x.dim() != 3:
        #     raise ValueError(f"Expected input of shape (batch_size, seq_len, d_model), got {x.shape}")
        
        batch_size, seq_len, d_model = x.shape
        # if d_model != self.d_model:
        #     raise ValueError(f"Input feature dimension {d_model} doesn't match expected dimension {self.d_model}")
        
        # # Check for NaN values in input
        # if torch.isnan(x).any():
        #     raise ValueError("Input contains NaN values")
        
        # Linear projections
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # Apply KV compression if enabled
        if self.kv_compression is not None:
            # Apply compression to the sequence length dimension
            # Transpose to get (batch_size, d_model, seq_len)
            K = K.transpose(1, 2)  # (batch_size, d_model, seq_len)
            V = V.transpose(1, 2)  # (batch_size, d_model, seq_len)
            
            # Apply compression
            K = self.k_compression(K)  # (batch_size, d_model, compressed_seq_len)
            V = self.v_compression(V)  # (batch_size, d_model, compressed_seq_len)
            
            # Transpose back to (batch_size, compressed_seq_len, d_model)
            K = K.transpose(1, 2)  # (batch_size, compressed_seq_len, d_model)
            V = V.transpose(1, 2)  # (batch_size, compressed_seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with improved numerical stability
        # Calculate scaling factor safely
        scaling_factor = float(self.d_k ** 0.5)
        if scaling_factor == 0:
            scaling_factor = 1.0  # Safety check to avoid division by zero
            
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scaling_factor
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply numerical stability technique for softmax
        scores_max, _ = torch.max(scores, dim=-1, keepdim=True)
        scores = scores - scores_max  # Subtract max for numerical stability
        
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Check for NaN values
        if torch.isnan(attn).any():
            # Replace NaN values with zeros and normalize
            attn = torch.nan_to_num(attn, nan=0.0)
            # Ensure rows sum to 1
            attn_sum = attn.sum(dim=-1, keepdim=True)
            attn_sum = torch.where(attn_sum == 0, torch.ones_like(attn_sum), attn_sum)
            attn = attn / attn_sum
        
        # Apply attention weights to values
        context = torch.matmul(attn, V)
        
        # Reshape and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        # Final NaN check
        # if torch.isnan(output).any():
        #     raise ValueError("Output contains NaN values")
            
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1, activation="ReGLU"):
        super().__init__()
        if activation=="ReGLU":
            in_d_hidden = d_hidden * 2
        else:
            in_d_hidden = d_hidden
        self.linear1 = nn.Linear(d_model, in_d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = make_activation(activation)
        
    def forward(self, x):
        # Apply first linear layer and activation
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_hidden, dropout=0.1, kv_compression=None, 
                 prenormalization=True, attention_initialization="kaiming",
                 attention_normalization="LayerNorm", ffn_activation="ReGLU",
                 ffn_normalization="LayerNorm"):
        super().__init__()
        self.prenormalization = prenormalization
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, kv_compression, attention_initialization)
        self.feed_forward = FeedForward(d_model, d_hidden, dropout, ffn_activation)
        
        self.attention_norm = make_nn_module(attention_normalization, d_model)
        self.ffn_norm = make_nn_module(ffn_normalization, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        if self.prenormalization:
            # Pre-normalization
            # Check for NaN values
            # if torch.isnan(x).any():
            #     raise ValueError("Input to TransformerBlock contains NaN values")
            x_norm = self.attention_norm(x)
            # Check for NaN values
            # if torch.isnan(x_norm).any():
            #     raise ValueError("Input to Attention contains NaN values")
            attn_output = self.attention(x_norm, mask)

            # Check for NaN values
            # if torch.isnan(attn_output).any():
            #     raise ValueError("Attention output contains NaN values")
            
            x = x + self.dropout(attn_output)

            # Check for NaN values
            # if torch.isnan(x).any():
            #     raise ValueError("Post-attention output contains NaN values")
            
            x_norm = self.ffn_norm(x)
            # Check for NaN values
            # if torch.isnan(x_norm).any():
            #     raise ValueError("Input to FeedForward contains NaN values")


            ff_output = self.feed_forward(x_norm)
            # Check for NaN values
            # if torch.isnan(ff_output).any():
            #     raise ValueError("FeedForward output contains NaN values")
            
            x = x + self.dropout(ff_output)
        else:
            # Post-normalization
            attn_output = self.attention(x, mask)
            x = self.attention_norm(x + self.dropout(attn_output))
            
            ff_output = self.feed_forward(x)
            x = self.ffn_norm(x + self.dropout(ff_output))
        
        return x
    
class CLSToken(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # Shape: (1, 1, d_model)
        
    def expand(self, batch_size):
        return self.cls_token.expand(batch_size, -1, -1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Check for NaN values
        if torch.isnan(cls_tokens).any():
            raise ValueError("CLS token after expand contains NaN values")
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, seq_len + 1, d_model)
        # Check for NaN values
        if torch.isnan(x).any():
            raise ValueError("Input with CLS token contains NaN values")
        return x
    

class Transformer_Model(nn.Module):
    def __init__(self, 
                 n_tokens,
                 objective,
                 num_classes,
                 residual_dropout, 
                 n_blocks, 
                 attention_dropout, 
                 ffn_dropout, 
                 ffn_d_hidden, 
                 d_tokens,
                 prenormalization=True, 
                 first_prenormalization=False, 
                 attention_initialization="kaiming",
                 attention_normalization="LayerNorm", 
                 ffn_activation="ReGLU", 
                 ffn_normalization="LayerNorm",
                 head_activation="ReLU", 
                 head_normalization="LayerNorm", 
                 
                 last_layer_query_idx=None, 
                 kv_compression_ratio=None, 
                 kv_compression_sharing=None):
        super().__init__()
        self.n_tokens = n_tokens
        self.residual_dropout = residual_dropout
        self.n_blocks = n_blocks
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.ffn_d_hidden = ffn_d_hidden
        self.d_tokens = d_tokens
        self.prenormalization = prenormalization
        self.first_prenormalization = first_prenormalization
        self.last_layer_query_idx = last_layer_query_idx
        
        self.d_model = d_tokens
        self.n_heads = 8  # Fixed number of attention heads
        
        # KV compression settings
        if kv_compression_ratio is not None and kv_compression_sharing is not None:
            # Account for CLS token in the sequence length
            self.kv_compression = {
                'n_tokens': self.n_tokens + 1,  # +1 for CLS token
                'ratio': kv_compression_ratio,
                'sharing': kv_compression_sharing
            }
        else:
            self.kv_compression = None
        
        self.cls_token = CLSToken(self.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_hidden=self.ffn_d_hidden,
                dropout=self.attention_dropout,
                kv_compression=self.kv_compression,
                prenormalization=self.prenormalization if i > 0 or self.first_prenormalization else False,
                attention_initialization=attention_initialization,
                attention_normalization=attention_normalization,
                ffn_activation=ffn_activation,
                ffn_normalization=ffn_normalization
            ) for i in range(self.n_blocks)
        ])
        
        # Head
        self.head = nn.Sequential(
            make_nn_module(head_normalization, self.d_model),
            make_activation(head_activation),
            nn.Linear(self.d_model, 1 if objective == "regression" else num_classes)
        )
        
        # Dropout
        self.dropout = nn.Dropout(self.residual_dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size  = x.shape[0]
        
        # Check for NaN values
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values")

        if torch.isnan(self.cls_token).any():
            raise ValueError("CLS token before expand contains NaN values")
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Check for NaN values
        if torch.isnan(cls_tokens).any():
            raise ValueError("CLS token after expand contains NaN values")
        
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, seq_len + 1, d_model)
        # Check for NaN values
        if torch.isnan(x).any():
            raise ValueError("Input with CLS token contains NaN values")
        
        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Check for NaN values
            if torch.isnan(x).any():
                raise ValueError(f"Output of block {i} contains NaN values")
        
        # Use CLS token for prediction
        x = x[:, 0]  # (batch_size, d_model)
        
        # Apply head
        x = self.head(x)  # (batch_size, 1) or (batch_size, num_classes)

        # Check for NaN values
        if torch.isnan(x).any():
            raise ValueError("Output contains NaN values")
        
        return x

