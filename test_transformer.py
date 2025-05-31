import torch
from argparse import Namespace
from models.transformer_rtdl import Transformer
from models.basemodule_embedding import Tokenizer

def test_transformer():
    # Create test arguments
    from argparse import Namespace
    args = Namespace(
        num_bins_source="",
        num_bins=8,  # Token dimension
        data_loader='torch',
        batch_size=2,
        numerical_embedding=True,
        control_group=False,
        shuffle=False,
        seed=42,
        embedding_module='tokenizer',
        num_features=5,  # Total number of features
        num_idx=[0, 1],  # Indices of numerical features
        cat_dims=[9,3, 4],  # Cardinalities of categorical features
        use_gpu = False,
        data_parallel = False,
        activation="tanh",
        use_M_matrix=False,
        second_activation="identity",
        initialization="alpha",
        with_original_features=False,
        model_name="transformer",
        objective="regression",
    )
    params = {
            "residual_dropout": 0.25,
            "n_blocks": 4,
            "attention_dropout": 0.01,
            "ffn_dropout": 0.00,
            "ffn_d_hidden": 100,
            "num_bins": 8,
            "learning_rate": 0.001,
            "prenormalization": False,
            "first_prenormalization": False,
            "attention_initialization": "kaiming",
            "attention_normalization": "LayerNorm",
            "ffn_activation": "ReLU",
            "ffn_normalization": "LayerNorm",
            "head_activation": "ReLU",
            "head_normalization": "LayerNorm",
        }

    # Create transformer model
    transformer = Transformer(params,args)

    # Create sample input data
    n = 2
    x_num = torch.randn(n, len(args.num_idx))  # Numerical features
    x_cat = torch.cat([torch.randint(0, d, (n, 1)) for d in args.cat_dims], dim=1)  # Categorical features
    x = torch.cat([x_num, x_cat], dim=1)  # Combined features


    # Create sample target data (for regression)
    y = torch.randn(n, 1)

    # Test forward pass
    print("Input shape:", x.shape)
    transformer.fit(x,y,x,y)
    output = transformer.predict(x)
    print("Output shape:", output.shape)
    print("Output sample:", output[0])

if __name__ == "__main__":
    test_transformer() 