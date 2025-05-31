import torch
import torch.nn as nn
import math

PASSING_ORIGIN = False # Only effective for NumericalEmbedding2

def adjust_input_dim(args):
    input_dim = args.num_features
    num_idx = args.num_idx
    num_bins = args.num_bins
    if PASSING_ORIGIN:
        return input_dim + len(num_idx) * (num_bins)
    else:
        return input_dim + len(num_idx) * (num_bins-1)

class NumEmbWrapper(nn.Module):
    def __init__(self,model,params,args):
        super(NumEmbWrapper, self).__init__()
        self.main_model = model
        print(args.cat_idx,args.num_idx)
        self.num_emb = NumericalEmbedding(args.num_features, args.num_bins, #params['num_bins'],
                                           activation=self.str2act(args.activation),
                                           use_M_matrix=args.use_M_matrix,num_idx=args.num_idx,
                                           double_activation=args.double_activation,
                                           initialization=args.initialization)
        # self.num_emb = NumericalEmbedding2(num_bins = params['num_bins'],
        #                                    activation=self.str2act(args.activation),
        #                                    use_M_matrix=args.use_M_matrix,num_idx=args.num_idx,
        #                                    double_activation=args.double_activation,
        #                                    initialization=args.initialization)
        #print(f"Numerical embedding is used: {params['num_bins']}")

    def forward(self, x):
        x = self.num_emb(x)
        x = self.main_model(x)
        return x
    
    def str2act(self,actstr):
        if actstr == "tanh":
            return nn.Tanh()
        elif actstr == "relu":
            return nn.ReLU()
        elif actstr == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError("Activation function not available")
    
class NumericalEmbedding(nn.Module):
    '''
    Apply a binning transformation to each numerical column. 
    cat_idx: indicator on which columns are categorical.
    num_features: number of all features.
    num_bins: number of bins to use for each numerical column.
    use_M_matrix: whether to use a M matrix to transform the output of the binning layer.
    '''
    class Bin(nn.Module):
        '''
        Intake a column and apply a binning transformation.
        '''
        def __init__(self, num_bins, activation=nn.Tanh(), use_M_matrix=False, double_activation = False, initialization='uniform'):
            super(NumericalEmbedding.Bin, self).__init__()
            self.bin = nn.Linear(1, num_bins) 
            self.activation = activation
            self.use_M_matrix = use_M_matrix
            self.M_matrix = nn.Linear(num_bins, num_bins, bias=False) if use_M_matrix else None 
            self.double_activation = double_activation

            if initialization == "uniform":
                pass # default initialization
            elif initialization == "alpha":
                nn.init.uniform_(self.bin.bias, -num_bins, 0) 
                ###  weight x - bias = weight ( x - bias/weight) 
                # if x is with in [0,1] (minmax scaler)
                # if x is with [-3,3] (gaussian scaler)
                nn.init.constant_(self.bin.weight, num_bins)
                if use_M_matrix:
                    nn.init.kaiming_uniform_(self.M_matrix.weight) # should I use this or random matrix?
            elif initialization == "unified":
                multiplier = math.sqrt(num_bins)
                nn.init.uniform_(self.bin.bias, -1/multiplier, 1/multiplier)
                nn.init.uniform_(self.bin.weight, -1/multiplier, 1/multiplier)
                if use_M_matrix:
                    nn.init.kaiming_uniform_(self.M_matrix.weight) # should I use this or random matrix?
            else:
                raise ValueError("Initialization method not available")

        def forward(self, x):
            x = self.bin(x)
            x = self.activation(x)
            if self.use_M_matrix:
                x = self.M_matrix(x)
            if self.double_activation:
                x = nn.ReLU()(x)
            return x    
        
    def __init__(self, num_features, num_bins, use_M_matrix=False, num_idx=[], activation=nn.Tanh(), double_activation =False, initialization='uniform'):
        super(NumericalEmbedding, self).__init__()
        self.num_idx = num_idx
        self.bins = nn.ModuleList([NumericalEmbedding.Bin(num_bins, use_M_matrix=use_M_matrix, activation=activation, initialization=initialization,double_activation=double_activation) if i in self.num_idx else None for i in range(num_features)])
        self.activation = activation

    def forward(self, x):
        return torch.cat([self.bins[i](x[:,i:i+1]) if i in self.num_idx else x[:,i:i+1] for i in range(x.shape[1])],dim=1)
        #column_outputs = [x[:,i:i+1] for i in range(x.shape[1])]
        #for i in self.num_idx:
        #    column_outputs[i] = self.bins[i](column_outputs[i])
        #return torch.cat(column_outputs, dim=1)

class NumericalEmbedding2(nn.Module):          
    def __init__(self, num_bins, use_M_matrix=False, num_idx=[], activation=nn.Tanh(), double_activation =False, initialization='uniform'):
        super().__init__()
        self.num_idx = num_idx
        self.activation = activation
        self.use_M_matrix = use_M_matrix
        self.double_activation = double_activation
        num_numerical_features = len(num_idx)
        self.weight = nn.Parameter(torch.Tensor(num_numerical_features, num_bins))
        self.bias = nn.Parameter(torch.Tensor(num_numerical_features, num_bins)) 
        self.initialize_parameters(initialization,num_bins)
        #self.print_parameters()
        
    def print_parameters(self):
        for i in range(len(self.num_idx)):
            print(f"weight and bias for feature {self.num_idx[i]} : ", self.weight[i,:].detach().numpy(),self.bias[i,:].detach().numpy())

    def initialize_parameters(self,initialization,num_bins):
        if initialization == "uniform":
            nn.init.uniform_(self.weight, -1, 1)
            nn.init.uniform_(self.bias,-1,1)
        elif initialization == "unified":
            multiplier = math.sqrt(num_bins)
            nn.init.uniform_(self.weight, -1/multiplier, 1/multiplier)
            nn.init.uniform_(self.bias,-1/multiplier,1/multiplier)
        elif initialization == "alpha":
            nn.init.uniform_(self.bias, -num_bins, 0)
            nn.init.constant_(self.weight, num_bins)
        else:
            raise ValueError("Initialization method not available")
        

    def forward(self, x):
        x_num = x[:,self.num_idx]
        x_num_copy = x_num.clone() if PASSING_ORIGIN else None
        x_cat = x[:,[i for i in range(x.shape[1]) if i not in self.num_idx]]
        x_num = self.weight[None,...] * x_num[..., None]
        #print(x.shape)
        x_num = x_num + self.bias[None,...]
        x_num = self.activation(x_num)
        x_num = x_num.view(x.size(0), -1)
        #if self.use_M_matrix:
        #    x = torch.matmul(x, self.M_matrix)
        #if self.double_activation:
        #    x = nn.ReLU()(x)
        if PASSING_ORIGIN:
            return torch.cat((x_cat,x_num,x_num_copy),dim=1)
        else:
            return torch.cat((x_cat,x_num),dim=1)


if __name__ == "__main__":
    x1 = torch.tensor([[(i+1)*(j+1) for i in range(2)] for j in range(5)]).float()
    x2 = torch.tensor([[3*i] for i in range(5)])
    x = torch.cat([x1, x2], dim=1)
    model2 = NumericalEmbedding2(num_bins=3, num_idx=[1,2],initialization='alpha',activation=nn.Tanh())
    model1 = NumericalEmbedding(num_features=3, num_bins=3, num_idx=[1,2],initialization='alpha',activation=nn.Tanh())
    print(model1(x),model2(x))

    # class ARGS:
    #     def __init__(self):
    #         self.num_bins = 3
    #         self.use_M_matrix = True
    #         self.num_idx = [0]
    # model = model.to('cuda')
    # arg = ARGS()
    # model = NumEmbWrapper(model,args=arg)