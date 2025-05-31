import sklearn.datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
import os
import certifi
import ssl
from pathlib import Path
import numpy as np
from urllib.request import urlretrieve
import pandas as pd


def one_hot_encoding(X, args):
    print("OneHotEncoding the data...")
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_cat = ohe.fit_transform(X[:, args.cat_idx])
    X = np.concatenate([X_cat, X[:, args.num_idx]], axis=1)

    args.num_features = X.shape[1]
    args.cat_idx = list(range(X_cat.shape[1]))
    args.cat_dims = [1] * len(X_cat.shape[1])
    
    args.num_idx = list(range(X_cat.shape[1], X.shape[1]))
    print("New Shape:", X.shape)
    return X

def discretize_colum(data_clm, num_values=10):
    """ Discretize a column by quantiles """
    r = np.argsort(data_clm)
    bin_sz = (len(r) / num_values) + 1  # make sure all quantiles are in range 0-(num_quarts-1)
    q = r // bin_sz
    return q

def safe_download_with_cache(url: str, cache_dir: Path) -> Path:
    """Download a file with robust error handling and caching"""
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache filename from URL
    cache_file = cache_dir / f"{hash(url)}.cache"
    
    if cache_file.exists():
        return cache_file
        
    # Set up SSL context
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    os.environ['SSL_CERT_FILE'] = certifi.where()
    
    try:
        print(f'Downloading {url} ...', end='', flush=True)
        urlretrieve(url, cache_file, context=ssl_context)
        return cache_file
    except Exception as e:
        if cache_file.exists():
            cache_file.unlink()
        raise RuntimeError(f"Failed to download {url}: {str(e)}")

def load_rtdl_dataset(args):
    """Load a dataset from the RTDL collection"""
    print("    Loading dataset " + args.dataset + " from RTDL library...")
    from .load_data_rtdl import (
        eye_movements, gesture_phase, house_16h, higgs_small,
        santander_customer_transactions, otto_group_products,
        churn_modelling, facebook_comments_volume, california_housing,
        covtype, adult, DATA_DIR
    )
    
    # Mapping of dataset names to their loading functions
    rtdl_datasets = {
        'eye': eye_movements,
        'gesture': gesture_phase,
        'house': house_16h,
        'higgs-small': higgs_small,
        'santander': santander_customer_transactions,
        'otto': otto_group_products,
        'churn': churn_modelling,
        'fb-comments': lambda: facebook_comments_volume(keep_derived=True),
        'california': california_housing,
        'covtype': covtype,
        'adult': adult
    }
    
    if args.dataset not in rtdl_datasets:
        raise ValueError(f"    RTDL dataset {args.dataset} not supported. Available datasets: {list(rtdl_datasets.keys())}")
    
    try:
        # Call the dataset function to save the files
        rtdl_datasets[args.dataset]()
        
        # Load the saved files
        dataset_dir = DATA_DIR / args.dataset
        data = {}
        
        # Load numerical features if they exist
        try:
            data['X_num'] = {
                'train': np.load(dataset_dir / 'X_num_train.npy'),
                'val': np.load(dataset_dir / 'X_num_val.npy'),
                'test': np.load(dataset_dir / 'X_num_test.npy')
            }
        except FileNotFoundError:
            data['X_num'] = None
            
        # Load categorical features if they exist
        try:
            data['X_cat'] = {
                'train': np.load(dataset_dir / 'X_cat_train.npy'),
                'val': np.load(dataset_dir / 'X_cat_val.npy'),
                'test': np.load(dataset_dir / 'X_cat_test.npy')
            }
        except FileNotFoundError:
            data['X_cat'] = None
            
        # Load target values
        data['y'] = {
            'train': np.load(dataset_dir / 'y_train.npy'),
            'val': np.load(dataset_dir / 'y_val.npy'),
            'test': np.load(dataset_dir / 'y_test.npy')
        }
        
        # Rest of the function remains the same
        parts = ['train', 'val', 'test']
        all_X = []
        all_y = []
        
        for part in parts:
            X_part = []
            if data['X_num'] is not None:
                X_part.append(data['X_num'][part])
            if data['X_cat'] is not None:
                X_part.append(data['X_cat'][part])
            
            X_combined = np.hstack(X_part)
            all_X.append(X_combined)
            all_y.append(data['y'][part])
        
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        # Update indices and dimensions
        if data['X_num'] is not None:
            n_num = data['X_num']['train'].shape[1]
            args.num_idx = list(range(n_num))
        else:
            n_num = 0
            args.num_idx = []
            
        args.cat_dims = []
        
        if data['X_cat'] is not None:
            X_cat_train = data['X_cat']['train']
            args.cat_idx = list(range(n_num, n_num + X_cat_train.shape[1]))
            
            for i in range(X_cat_train.shape[1]):
                unique_vals = set()
                for part in parts:
                    unique_vals.update(set(data['X_cat'][part][:, i]))
                args.cat_dims.append(len(unique_vals))
        else:
            args.cat_idx = []
        
        args.num_features = X.shape[1]
        
        print(f"    Dataset loaded with {len(args.num_idx)} numerical features and {len(args.cat_idx)} categorical features")
        if args.cat_dims:
            print(f"    Categorical dimensions: {args.cat_dims}")
        return X, y
        
    except Exception as e:
        raise RuntimeError(f"    Failed to load RTDL dataset {args.dataset}: {str(e)}")

def load_raw_data(args):
    if args.source.startswith('rtdl'):
        X, y = load_rtdl_dataset(args)
        
    elif args.dataset == "CaliforniaHousing":
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

    elif args.dataset == "Covertype":
        X, y = sklearn.datasets.fetch_covtype(return_X_y=True)
        
    elif args.dataset == "Adult" or args.dataset == "AdultCat":
        url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        label = "income"
        columns = features + [label]

        # Use safe download with caching
        cache_dir = Path('data/cache')
        data_file = safe_download_with_cache(url_data, cache_dir)
        df = pd.read_csv(data_file, names=columns)

        # Rest of your Adult dataset processing code...
        df.fillna(0, inplace=True)
        if args.dataset == "AdultCat":
            columns_to_discr = [('age', 10), ('fnlwgt', 25), ('capital-gain', 10), ('capital-loss', 10),
                              ('hours-per-week', 10)]
            for clm, nvals in columns_to_discr:
                df[clm] = discretize_colum(df[clm], num_values=nvals)
                df[clm] = df[clm].astype(int).astype(str)
            df['education_num'] = df['education_num'].astype(int).astype(str)
            args.cat_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        X = df[features].to_numpy()
        y = df[label].to_numpy()

    elif args.dataset == "HIGGS":  # Binary classification dataset with one categorical feature
        path = "/opt/notebooks/data/HIGGS.csv.gz"
        df = pd.read_csv(path, header=None)
        df.columns = ['x' + str(i) for i in range(df.shape[1])]
        num_col = list(df.drop(['x0', 'x21'], 1).columns)
        cat_col = ['x21']
        label_col = 'x0'

        def fe(x):
            if x > 2:
                return 1
            elif x > 1:
                return 0
            else:
                return 2

        df.x21 = df.x21.apply(fe)

        # Fill NaN with something better?
        df.fillna(0, inplace=True)

        X = df[num_col + cat_col].to_numpy()
        y = df[label_col].to_numpy()

    elif args.dataset == "Heloc":  # Binary classification dataset without categorical data
        path = "heloc_cleaned.csv"  # Missing values already filtered
        df = pd.read_csv(path)
        label_col = 'RiskPerformance'

        X = df.drop(label_col, axis=1).to_numpy()
        y = df[label_col].to_numpy()

    else:
        raise AttributeError("Dataset \"" + args.dataset + "\" not supported")

    print("    Raw Dataset loaded!")
    return X, y

def load_data(args):
    
    print("Loading dataset " + args.dataset + "...")

    X, y = load_raw_data(args)

    #print("    X[0]: ",X[0])
    #print("    y.unique.top(10): ",np.unique(y)[:10])
    # Using small samples for fast framework development
    if args.small_sample:
        print("Using only 1000 samples...")
        X, y = X[:1000, :], y[:1000]
    print(f"Dataset size: {X.shape}, {y.shape}")
    
    args.num_samples = X.shape[0]  
    args.num_features = X.shape[1]  #encoded in the config, but why?
    if args.cat_idx is None:
        args.cat_idx = []

    # Reorder categorical and numerical columns
    args.num_idx = sorted(list(set(range(args.num_features))-set(args.cat_idx)))
    X = X[:,args.num_idx + args.cat_idx]
    args.num_idx, args.cat_idx  = list(range(len(args.num_idx))), list(range(len(args.num_idx),args.num_features))
    if len(args.cat_idx) > 0:
        print(f"Reorder columns: cat from {min(args.cat_idx)} to {max(args.cat_idx)}")
    else:
        print("No categorical columns")

    # Encoding categorical data
    args.cat_dims = []  # cat_dims encoded in the config, but why?
    for i in args.cat_idx:
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])
        args.cat_dims.append(len(le.classes_))
    
    if args.one_hot_encode:
        one_hot_encoding(X, args)
    if args.target_encode:
        le = LabelEncoder()
        y = le.fit_transform(y)
        # Setting this if classification task
        if args.objective == "classification":
            args.num_classes = len(le.classes_)
            print(f"Having {args.num_classes} classes as target.")

    # Scaling numerical data
    if args.scale:
        if args.scaler == "standard":
            print("Scaling the data using standard scaler...")
            scaler = StandardScaler()
            X[:, args.num_idx] = scaler.fit_transform(X[:, args.num_idx])
            if args.scale_target and (args.objective == "regression"):
                y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
        elif args.scaler == "minmax":
            print("Scaling the data using minmax scaler...")
            scaler = MinMaxScaler(feature_range=(-1,1))
            X[:, args.num_idx] = scaler.fit_transform(X[:, args.num_idx])
            if args.scale_target and (args.objective == "regression"):
                y = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
        else:
            raise AttributeError(f"Scaler \"{args.scaler}\" not available")

    print("Dataset loaded!")
    return X.astype(np.float32), y

if __name__ == "__main__":
    from argparse import Namespace
    import yaml
    args = Namespace(small_sample=True, 
                    source='rtdl', 
                    one_hot_encode=True,
                    target_encode=True,
                    scale=True,
                    scaler='minmax',
                    scale_target=True)
    for dataset in ['otto','santander', 'churn', 'higgs-small', 'house', 'gesture', 'eye', 'fb-comments', 'california', 'covtype', 'adult']:
        args.dataset = dataset
        # read objective from config file in config/rtdl/
        with open(f'config/rtdl/{dataset}.yml', 'r') as f:
            config = yaml.safe_load(f)
        args.objective = config['objective']

        X, y = load_raw_data(args)
        #print("X[0]: ",X[0])
        print(args.dataset)
        print("y.unique().top(10): ",np.unique(y)[:10])
        print(args.objective)
        print(args.num_features)
        print(args.num_idx)
        print(args.cat_idx)
        print(args.cat_dims)
        

        # # Mapping of dataset names to their loading functions
        # rtdl_datasets = {
        #     'gesture': gesture_phase,
        #     'house': house_16h,
        #     'higgs-small': higgs_small,
        #     'santander': santander_customer_transactions,
        #     'otto': otto_group_products,
        #     'churn': churn_modelling,
        #     'fb-comments': lambda: facebook_comments_volume(keep_derived=True),
        #     'california': california_housing,
        #     'covtype': covtype,
        #     'adult': adult
        # }