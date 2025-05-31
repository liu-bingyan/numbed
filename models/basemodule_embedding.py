import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Any

def adjust_input_dim(args):
    if not args.numerical_embedding:
        return args.num_features
    
    input_dim = args.num_features
    num_idx = args.num_idx
    num_bins = args.num_bins

    if args.embedding_module == "tokenizer":
        return input_dim * num_bins   # in tokenizer, this is not explicitly used
    elif args.embedding_module == "numerical_embedding":
        if args.with_original_features:
            return input_dim + len(num_idx) * (num_bins)
        else:
            return len(num_idx) * (num_bins-1)
    else:
        raise ValueError("Invalid embedding module")
    
def select_embedding_module(args):
    non_flat_models = ["transformer"]
    if args.model_name in non_flat_models:
        emb_module = "tokenizer"
    else:
        emb_module = "numerical_embedding"
    print("Using embedding module: ", emb_module)
    return emb_module

def str2act(actstr):
        if actstr == "tanh":
            return nn.Tanh()
        elif actstr == "relu":
            return nn.ReLU()
        elif actstr == "sigmoid":
            return nn.Sigmoid()
        elif actstr == "identity":
            return nn.Identity()
        else:
            raise ValueError("Activation function not supported")
    
class ControlGroup(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.output_dim = adjust_input_dim(args)
        if args.second_activation != "":
            self.activation = str2act(args.second_activation)
        else:
            self.activation = str2act(args.activation)
        self.fc = nn.Linear(args.num_features, self.output_dim)
    def forward(self, x):
        return self.activation(self.fc(x))    

class NumericalEmbedding(nn.Module):
    def __init__(self, args, as_tokenizer = False):
        super().__init__()

        self.args = args
        self.as_tokenizer = as_tokenizer
        self.activation = str2act(args.activation)        
        if args.second_activation == "":
            self.second_activation = nn.Identity()
        else:
            self.second_activation = str2act(args.second_activation)
        
        # assuming data is sorted so that numerical features are before categorical features       
        self.k = len(args.num_idx)
        self.d = args.num_bins

        self.W = nn.Parameter(torch.ones(self.k, self.d))  # shape: (k, d)
        self.b = nn.Parameter(torch.randn(self.k, self.d))  # shape: (k, d)
        self.M = nn.Parameter(torch.randn(self.k, self.d, self.d)) if args.use_M_matrix else None # shape: (k, d, d)
        self.b2 = nn.Parameter(torch.randn(self.k, self.d)) if args.use_M_matrix else None # shape: (k, d)

        self.reset_parameters()

    def reset_parameters(self):
        if self.args.initialization == 'standard':
            print("Initializing numericalembeddings with standard methods")
            if self.args.activation == 'tanh' or self.args.activation == 'identity':
                gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(self.W, gain=gain)
                nn.init.uniform_(self.b, -1, 1)
            elif self.args.activation == 'relu':
                nn.init.kaiming_uniform_(self.W, a=0)
                nn.init.uniform_(self.b, -1, 1)
            # elif self.args.activation == 'sigmoid':
            #     nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('sigmoid'))
            #     nn.init.uniform_(self.b, -1, 1)
            else:
                raise ValueError("Activation function not supported")
        elif self.args.initialization == 'alpha':
            print("Initializing numerical embeddings with alpha initialization")
            if self.args.activation == 'tanh' or self.args.activation == 'identity': # our proposed initialization
                nn.init.constant_(self.W,self.d)
                nn.init.uniform_(self.b,-self.d,self.d)
            elif self.args.activation == 'relu': # test if our proposed initialization works for relu
                gain = nn.init.calculate_gain('relu')
                nn.init.constant_(self.W,self.d)
                nn.init.uniform_(self.b,-gain * self.d,gain * self.d)
            # elif self.args.activation == 'sigmoid':
            #     nn.init.constant_(self.W,self.d)
            #     nn.init.uniform_(self.b,-1,1)
            else:
                raise ValueError("Activation function not supported")
        elif self.args.initialization == 'rtdl':
            s = self.d ** -0.5
            print(f"Initializing numerical embeddings with rtdl initialization s={s}")
            nn.init.uniform_(self.W, -s, s)
            nn.init.uniform_(self.b, -s, s)
        else:
            raise ValueError("Initialization method not supported")

        if self.args.use_M_matrix:
            if self.args.second_activation == 'tanh' or self.args.second_activation == 'identity':
                gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(self.M, gain=gain)
                nn.init.uniform_(self.b2, -1, 1)
            elif self.args.second_activation == 'relu':
                nn.init.kaiming_uniform_(self.M)
                nn.init.uniform_(self.b2, -1, 1)
            # elif self.args.second_activation == 'sigmoid':
            #     nn.init.xavier_uniform_(self.M, gain=nn.init.calculate_gain('sigmoid'))
            #     nn.init.uniform_(self.b2, -1, 1)
            else:
                raise ValueError("Activation function not supported")

    def forward(self, x):
        # x: (n, p)
        if self.as_tokenizer:
            x_num = x
            x_cat = None
        else:
            x_num = x[:, :self.k]             # (n, k)
            x_cat = x[:, self.k:]            # (n, p-k)

        # compute embedding for numerical features
        x_num_3d = x_num.unsqueeze(-1)       # (n, k, 1)
        x_num_emb = self.activation(x_num_3d * self.W + self.b) # (n, k, d)
        
        def transform(x_k, M_k, b2_k):
            return x_k @ M_k + b2_k
        if self.args.use_M_matrix:
           x_num_emb = torch.vmap(transform, in_dims=(1, 0, 0), out_dims=1)(x_num_emb, self.M, self.b2)  # (n, k, d)
           x_num_emb = self.second_activation(x_num_emb) # (n, k, d)

        if self.as_tokenizer:
            return x_num_emb
        else:
            x_num_emb = x_num_emb.reshape(x.size(0), -1) # (n, k*d)
            components = [x_num_emb]
            if self.args.with_original_features:
                components.append(x_num)
            components.append(x_cat)
            return torch.cat(components,dim=1)  # (n, k*d + p-k)
        
class CategoricalEmbedding(nn.Module):
    """Transforms categorical features to tokens (embeddings) following the NumericalEmbedding interface."""
    
    def __init__(self, args, as_tokenizer=False):
        """
        Args:
            args: Arguments containing initialization parameters
            as_tokenizer: If True, returns 3D tensor (batch_size, n_features, d_token)
                         If False, returns 2D tensor (batch_size, n_features * d_token)
        """
        super().__init__()
        self.args = args
        self.as_tokenizer = as_tokenizer

        # Extract parameters from args
        self.cat_dims = args.cat_dims
        self.k = len(args.num_idx)
        self.kc = len(args.cat_dims) # p-k
        self.d = args.num_bins
        self.initialization = args.initialization
        
        # Validate parameters        
        lookup_idx = torch.tensor([0] + self.cat_dims[:-1]).cumsum(0)
        self.register_buffer('lookup_idx', lookup_idx, persistent=False)
        
        self.E = nn.Embedding(sum(self.cat_dims), self.d)
        self.bc = nn.Parameter(torch.Tensor(self.kc, self.d))
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.initialization in ['standard', 'alpha', 'rtdl']:
            s = self.d**-0.5
            nn.init.uniform_(self.E.weight, -s, s)
            nn.init.uniform_(self.bc, -s, s)
        else:
            raise ValueError("Initialization method not supported")
        
    def forward(self, x):
        """
        Forward pass of the categorical embedding.
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
               Categorical features should be in the range [0, cardinality-1]
            
        Returns:
            If as_tokenizer=True: Tensor of shape (batch_size, n_cat_features, d_token)
            If as_tokenizer=False: Tensor of shape (batch_size, n_cat_features * d_token)
        """
        # x: (n, p)

        # Extract categorical features (assuming they come after numerical features)
        if self.as_tokenizer:
            x_num = None
            x_cat = x
        else:
            x_num = x[:, :self.k] # (n, k)
            x_cat = x[:, self.k:] # (n, p-k)
        # turn x_cat as integer tensor
        x_cat_emb = self.E(x_cat.to(torch.int64) + self.lookup_idx.unsqueeze(0)) # (n, p-k, d)
        x_cat_emb = x_cat_emb + self.bc.unsqueeze(0) # (n, p-k, d)
        
        if self.as_tokenizer:
            return x_cat_emb  
        else:
            x_cat_emb = x_cat_emb.reshape(x.size(0), -1)
            return torch.cat([x_num, x_cat_emb], dim=1)

class Tokenizer(nn.Module):  
    def __init__(self,args):
        """
        Args:
            n_num_features: the number of continuous features. Pass :code:`0` if there
                are no numerical features.
            cat_cardinalities: the number of unique values for each feature. See
                `CategoricalFeatureTokenizer` for details. Pass an empty list if there
                are no categorical features.
            d_token: the size of one token.
        """
        super().__init__()
        self.args = args
        self.k = len(args.num_idx)
        self.num_tokenizer = NumericalEmbedding(args, as_tokenizer=True)
        self.cat_tokenizer = CategoricalEmbedding(args, as_tokenizer=True)
    
    def forward(self, x):
        x_num = x[:, :self.k]
        x_cat = x[:, self.k:]
        x_num_emb = self.num_tokenizer(x_num)
        x_cat_emb = self.cat_tokenizer(x_cat)
        return torch.cat([x_num_emb, x_cat_emb], dim=1)


class EmbeddingWrapper(nn.Module):
    def __init__(self,model,args):
        super().__init__()
        self.main_model = model
        if args.control_group:
            self.num_emb = ControlGroup(args)
        elif args.embedding_module == "tokenizer":
            self.num_emb = Tokenizer(args)
        else:
            self.num_emb = NumericalEmbedding(args)

    def forward(self, x):
        x = self.num_emb(x)
        x = self.main_model(x)
        return x
    

def test_num_emb():
    # test code:
    from argparse import Namespace
    args = Namespace(numerical_embedding = True, control_group = False, num_features=3, num_idx=[0,1], num_bins=3, activation="tanh", 
                     use_M_matrix=False, second_activation="identity", initialization="standard", embedding_module="numerical_embedding",
                     with_original_features=True)
    model = nn.Linear(9,8)
    model2 = EmbeddingWrapper(model, args)
    x = torch.tensor([[1,1,1]],dtype=torch.float32)
    x1 = model2.num_emb(x)
    print(x.shape,x)
    print(x1.shape,x1)
    d1 = model2.num_emb.state_dict()
    print(d1['W'])
    print(d1['b'])
    #xmid = d1['W'] + d1['b']
    #M = d1['M']
    #b2 = d1['b2']
    #print(xmid[0,...] @ M[0,...] + b2[0,...])
    #print(xmid[1,...] @ M[1,...] + b2[1,...])
    print(x1.max(),x1.min())
    print(x1.mean(),x1.std())
    
def test_cat_emb():
    from argparse import Namespace
    args = Namespace(numerical_embedding = True, control_group = False, embedding_module = 'tokenizer',
                     num_features=3, num_idx=[0], cat_dims=[3,4], num_bins=3, activation="identity", 
                     use_M_matrix=True, second_activation="identity", initialization="rtdl",
                     with_original_features=False )
    ce = Tokenizer(args)
    # createa a tensor with first len(num_idx) features being numerical and last len(cat_dims) being categorical
    n = 2
    x_num = torch.randn(n, len(args.num_idx))
    x_cat = torch.cat([torch.randint(0, d, (n,1)) for d in args.cat_dims], dim=1)
    x = torch.cat([x_num, x_cat], dim=1)
    x_emb = ce(x)
    print(x)
    print(x_emb.shape)
    print(x_emb)
    d1 = ce.state_dict()
    print("E. weight:")
    print(d1['E.weight'])
    print("bc:")
    print(d1['bc'])

def test_tokenizer():
    from argparse import Namespace
    args = Namespace(numerical_embedding = True, control_group = False, embedding_module = 'tokenizer',
                     num_features=3, num_idx=[0], cat_dims=[3,4], num_bins=3, activation="tanh", 
                     use_M_matrix=False, second_activation="identity", initialization="rtdl",
                     with_original_features=False )
    ce = Tokenizer(args)
    n = 2
    x_num = torch.randn(n, len(args.num_idx))
    x_cat = torch.cat([torch.randint(0, d, (n,1)) for d in args.cat_dims], dim=1)
    x = torch.cat([x_num, x_cat], dim=1)
    x_emb = ce(x)
    print(x)
    print(x_emb.shape)
    print(x_emb)

    d1 = ce.state_dict()
    print("W")
    print(d1['num_tokenizer.W'])
    print("b")
    print(d1['num_tokenizer.b'])
    #print("M")
    #print(d1['num_tokenizer.M'])
    #print("b2")
    #print(d1['num_tokenizer.b2'])
    print("E. weight")
    print(d1['cat_tokenizer.E.weight'])
    print("bc")
    print(d1['cat_tokenizer.bc'])

def test_embedding_wrapper():
    from argparse import Namespace
    args = Namespace(numerical_embedding = True, control_group = False, embedding_module = 'tokenizer',
                     num_features=3, num_idx=[0], cat_dims=[3,4], num_bins=3, activation="tanh", 
                     use_M_matrix=False, second_activation="identity", initialization="alpha",
                     with_original_features=False )
    model = nn.Linear(9,8)
    model2 = EmbeddingWrapper(model, args)
    x = torch.tensor([[1,1,1]],dtype=torch.float32)
    x_emb = model2(x)
    print(x_emb.shape)
    print(x_emb)

if __name__ == "__main__":
    test_num_emb()