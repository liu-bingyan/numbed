import logging
import sys
import threading
import time

import optuna

from models import str2model
from utils.load_data import load_data
from utils.scorer import get_scorer
from utils.timer import Train_Timer, Test_Timer
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file, save_results_to_json_file
from utils.parser import get_parser, get_given_parameters_parser
from sklearn.model_selection import KFold, StratifiedKFold  # , train_test_split


def get_scorer_timer_results(sc,train_timer,test_timer,args):
    scores = sc.get_results()
    timers = train_timer.get_results() | test_timer.get_results()
    dicts = {"scores" : scores , "timers" : timers, "args" : vars(args)}
    return dicts

def print_score_and_time(sc, t1, t2):
    print("Results:", sc.get_results())
    print("Train time:", t1.get_average_time())
    print("Inference time:", t2.get_average_time())

def cross_validation(model, X, y, args):

    sc = get_scorer(args)
    train_timer = Train_Timer(args)
    test_timer = Test_Timer(args)

    if args.objective == "regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "classification" or args.objective == "binary":
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        curr_model = model.clone()

        train_timer.start()
        try:
            loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_test, y_test)
        except RuntimeError as e:
            print(f"Memory state when error occurred:")
            # log_memory_usage()
            raise e
            
        train_timer.end(len(loss_history))

        if args.save_loss:
            save_loss_to_file(args, loss_history, "loss", extension=i)
            save_loss_to_file(args, val_loss_history, "val_loss", extension=i)
    
        # Test model
        test_timer.start()
        curr_model.predict(X_test)
        test_timer.end()
        sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)
        print(sc.get_results())

        curr_model.save_model_and_predictions(y_test, str(i))
        if args.numerical_embedding and args.save_embedded_features:
            curr_model.save_embedded_features(X, y, str(i))


    if args.save_results:
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), 
                             test_timer.get_average_time(),
                             model.params)
        
    if args.save_results_json:
        scores_dict = get_scorer_timer_results(sc, train_timer, test_timer, args)
        save_results_to_json_file(args, resultsname="scores", jsondict=scores_dict, append=False)
        
    return sc, train_timer, test_timer

class Objective(object):
    def __init__(self, args, model_name, X, y):
        # Save the model that will be trained
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y

        self.args = args

    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.args)

        # Create model
        model = self.model_name(trial_params, self.args)

        # Cross validate the chosen hyperparameters
        sc, time = cross_validation(model, self.X, self.y, self.args)

        save_hyperparameters_to_file(self.args, trial_params, sc.get_results(), time)

        return sc.get_objective_result()

def main(args):
    print("Start hyperparameter optimization")
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.model_name + "_" + args.dataset
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction=args.direction,
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    study.optimize(Objective(args, model_name, X, y), n_trials=args.n_trials)
    print("Best parameters:", study.best_trial.params)

    # Run best trial again and save it!
    model = model_name(study.best_trial.params, args)
    cross_validation(model, X, y, args)

def main_once(args):
    parameters = args.parameters[args.dataset][args.model_name_with_embedding_suffix]
    print("Train model with given hyperparameters:")
    print(parameters)

    X, y = load_data(args)
     
    print("Data loaded with the following dimensions:")
    print("num_idx: ", args.num_idx)
    print("cat_idx: ", args.cat_idx, "cat_dims: ", args.cat_dims)

    model_name = str2model(args.model_name)
    model = model_name(parameters, args)

    print("Updated arguments:")
    args.parameters = None
    print(args)

    sc,t1,t2 = cross_validation(model, X, y, args) 
    print_score_and_time(sc, t1, t2)
    

if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    print("Global arguments:")
    print(arguments)

    if arguments.optimize_hyperparameters:
        main(arguments)
    else:
        # Also load the best parameters
        parser = get_given_parameters_parser(arguments)
        arguments = parser.parse_args()
        print(f"Given parameters: {arguments.best_params_file}:")
        main_once(arguments)
