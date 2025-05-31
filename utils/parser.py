import configargparse
import yaml
import json


def get_parser():
    parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser,
                                           formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    # config files
    parser.add('-config', '--config', required=False, is_config_file_arg=True, help='config file path',
               default="config/TabSurvey/adult.yml")  # kddcup99 covertype california_housing adult higgs
    parser.add('-emb_config', '--emb_config', required=False, is_config_file_arg=True, help='training config file path',
               default="config/model_default.yml")  
    parser.add('-train_config', '--train_config', required=False, is_config_file_arg=True, help='io config file path',
               default="config/config_debug.yml")
    parser.add('--small_sample', action="store_true", help="Use only a small subset of the data for testing purposes")

    # model parameters
    parser.add('--model_name', required=False, default="MLP", help="Name of the model that should be trained")
    parser.add('--dataset', required=False, default="Adult", help="Name of the dataset that will be used")
    parser.add('--objective', required=True, type=str, default="regression", choices=["regression", "classification","binary"],
                help="Set the type of the task")    

    # gpu parameters
    parser.add('--use_gpu', action="store_true", help="Set to true if GPU is available")
    parser.add('--gpu_ids', type=int, action="append", help="IDs of the GPUs used when data_parallel is true")
    parser.add('--data_parallel', action="store_true", help="Distribute the training over multiple GPUs")

    # hyperparameter parameters
    parser.add('--optimize_hyperparameters', action="store_true",
               help="Search for the best hyperparameters")
    parser.add('--n_trials', type=int, default=5, help="Number of trials for the hyperparameter optimization")
    parser.add('--direction', type=str, default="minimize", choices=['minimize', 'maximize'],
               help="Direction of optimization.")
    parser.add('--num_splits', type=int, default=5, help="Number of splits done for cross validation")
    parser.add('--shuffle', action="store_true", help="Shuffle data during cross-validation")
    parser.add('--seed', type=int, default=123, help="Seed for KFold initialization.")

    # data parameters
    parser.add('--scale', action="store_true", help="Normalize input data.")
    parser.add('--target_encode', action="store_true", help="Encode the targets that they start at 0. (0, 1, 2,...)")
    parser.add('--one_hot_encode', action="store_true", help="OneHotEncode the categorical features")
    parser.add('--scale_target', action = "store_true", help="Scale the labels of the dataset.")
    parser.add('--scaler', type=str, default="standard", choices=['standard', 'minmax'],
               help="Scaler used for numerical features.")
    parser.add('--num_features', type=int, help="Set the total number of features. Will be set automatically when the load_data function is used.")
    parser.add('--num_classes', type=int, default=1, help="Set the number of classes in a classification task.")
    parser.add('--cat_idx', type=int, action="append", help="Indices of the categorical features")
    parser.add('--cat_dims', type=int, default = [], action="append", help="Cardinality of the categorical features (is set "
                                                             "automatically, when the load_data function is used.")

    # training parameters
    parser.add('--batch_size', type=int, default=128, help="Batch size used for training")
    parser.add('--val_batch_size', type=int, default=128, help="Batch size used for training and testing")
    parser.add('--data_loader',type = str, default="fast", help="Data loader to uses. Options: torch, fast")
    parser.add('--early_stopping_rounds', type=int, default=20, help="Number of rounds before early stopping applies.")
    parser.add('--epochs', type=int, default=1000, help="Max number of epochs to train.")
    parser.add('--logging_period', type=int, default=100, help="Number of iteration after which validation is printed.")

    
    
    # embedding parameters
    parser.add('--source', type=str, default="TabSurvey", choices=["TabSurvey", "rtdl", "rtdl-modelwise"], help="Source of the optimal parameters. \
               TabSurvey: TabSurvey, rtdl: RTDL with best default parameters, rtdl_modelwise: RTDL with best modelwise parameters")
    parser.add('--numerical_embedding', action="store_true", help="Use a numerical embedding layer for the numerical features.")
    parser.add('--control_group', action="store_true", help="Use a numerical embedding layer for the numerical features.")
    parser.add('--embedding_module', type=str, default="numerical_embedding", choices=["numerical_embedding", "tokenizer"], help="Type of the embedding module.")
    
    parser.add('--num_bins_source', type=str, default="", help="Name of the model to compare with.")
    parser.add('--num_bins', type=int, default=0, help="Number of bins to use for the numerical embedding layer.")
    
    parser.add('--activation', type=str, default="tanh", help="Activation function used in the numerical embedding layer.")
    parser.add('--use_M_matrix', action="store_true", help="Use a M matrix to transform the output of the binning layer.")
    parser.add('--second_activation', type=str, default="", help="Activation function used in the numerical embedding layer.")
    parser.add('--initialization', type=str, default="standard", choices = ['standard', 'alpha','rtdl'],  help="Initialization method for the numerical embedding layer.")
    parser.add('--with_original_features', action="store_true", help="Use the original features in the model.")
    
    # IO parameters
    parser.add('--save_model', action="store_true", help="Save the model after training.")
    parser.add('--record_embedding_parameters', action="store_true", help="Record the parameters in the embedding layers.")
    parser.add('--save_predictions', action="store_true", help="Save the predictions of the test dataset.")
    parser.add('--save_loss', action="store_true", help="Save the loss of the training.")
    
    parser.add('--save_results_json', action="store_true", help="Save the scores of the training.")
    parser.add('--save_results', action="store_true", help="Save the results after training.")
    parser.add('--save_embedded_features', action="store_true", help="Save the embedded data after training.")


    

  

    # Todo: Validate the arguments
    return parser

# def get_numerical_embedding_parser():
#     parser = get_parser()

#     parser.add('--num_bins', type=int, default=3, help="Number of bins to use for the numerical embedding layer.")
#     parser.add('--activation', type=str, default="tanh", help="Activation function used in the numerical embedding layer.")
#     parser.add('--use_M_matrix', action="store_true", help="Use a M matrix to transform the output of the binning layer.")
#     parser.add('--double_activation', action="store_true", help="Use an additional activation function after the M matrix.")
#     parser.add('--initialization', type=str, default="uniform", help="Initialization method for the numerical embedding layer.")

#     return parser
def get_model_name_with_embedding_suffix(args):
    if args.source != 'rtdl-modelwise':
        return args.model_name
    
    if args.num_bins_source == "":
        return args.model_name
    
    name = "-".join([args.model_name]+args.num_bins_source.split('-')[1:])
    return name


def get_given_parameters_parser(args):
    parser = get_parser()

    if args.source == 'TabSurvey':
        default_params = f"config/{args.source}/best_params.yml"
        load_method = yaml.safe_load
    elif args.source == 'rtdl':
        default_params = f"config/{args.source}/best_params.json"
        load_method = json.loads
    elif args.source == 'rtdl-modelwise':
        default_params = f"config/rtdl/best_params_modelwise.json"
        load_method = json.loads

    parser.add('-model_name_with_embedding_suffix', '--model_name_with_embedding_suffix', type=str,
               default=get_model_name_with_embedding_suffix(args), help="Name of the model that should be trained")
    parser.add('-best_params_file', '--best_params_file', is_config_file_arg=True, default=default_params,
               help="Parameter file path")
    parser.add('-parameters', '--parameters', type=yaml.safe_load, help="Parameter values")
    return parser


def get_attribution_parser():
    # Use parser that can read YML files
    parser = get_parser()

    parser.add('-paramsfile', '--paramsfile', required=False, is_config_file_arg=True, help='parameter file path',
               default="config/adult_params.yml")  # kddcup99 covertype california_housing adult higgs

    parser.add('-parameters', '--parameters', type=yaml.safe_load, help='parameter values')

    parser.add('--globalbenchmark', action="store_true", help="Run a ablation global attribution benchmark.")
    parser.add('--compareshap', action="store_true", help="Compare attributions to shapley values.")
    parser.add('--strategy', type=str, help="attribution computation strategy string")
    parser.add('--numruns', type=int, help="number of repetitions to run", default=1)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.source = "rtdl-modelwise"
    args.num_bins_source = "model-t-lr"
    args.model_name = "transformer"
    parser = get_given_parameters_parser(args)
    args = parser.parse_args()
    print(args.model_name_with_embedding_suffix)
