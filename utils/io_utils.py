import numpy as np
import os
import json
import pickle
import datetime
from argparse import Namespace

def output_dir(args):
    output_dir_name = f"output_{args.source}"
    if args.small_sample:
        output_dir_name += "_test/"
    else:
        output_dir_name += "/"
    return output_dir_name

def get_embedding_suffix(args):
    if not args.numerical_embedding:
        suffix = "default"
        return 
    elif args.control_group:
        suffix = "ctrl"
    else:
        suffix = "emb"

    suffix += "_" + str(args.num_bins)

    nbs = args.num_bins_source
    if nbs != '':
        suffix += "".join(str(nbs).removeprefix("model").split("-"))
    else:
        suffix += "p"

    suffix += "x" if args.with_original_features else ""
    suffix += "_" + args.activation

    if args.control_group:
        return suffix

    if args.use_M_matrix:
        suffix += f"_M"
        if args.second_activation != "none":
            suffix += f"_{str(args.second_activation)}"
    suffix += "_" + args.initialization
    return suffix

def get_output_path(args, filename, file_type, directory=None, extension=None):
    # For example: output/LinearModel/Covertype
    
    dir_path = output_dir(args) + args.model_name + "/" + args.dataset + "/" + get_embedding_suffix(args)

    if directory:
        # For example: .../models
        dir_path = dir_path + "/" + directory

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + "/" + filename

    if extension is not None:
        file_path += "_" + str(extension)

    file_path += "." + file_type

    # For example: .../m_3.pkl

    return file_path

def save_loss_to_file(args, arr, name, extension=""):
    filename = get_output_path(args, directory="logging", filename=name, extension=extension, file_type="txt")
    print("Saving loss to: ", filename)
    np.savetxt(filename, arr)
    np.savetxt(filename.replace(".txt", ".csv"), arr, fmt="%.6f", delimiter=',')

def save_predictions_to_file(arr, args, extension=""):
    filename = get_output_path(args, directory="predictions", filename="p", extension=extension, file_type="npy")
    print("Saving predictions to: ", filename)
    np.save(filename, arr)
    np.savetxt(filename.replace(".npy", ".csv"), arr, fmt="%.2f", delimiter=',')

def get_predictions_from_file(args):
    dir_path = output_dir(args) + args.model_name + "/" + args.dataset + "/predictions"

    files = os.listdir(dir_path)
    content = []

    for file in files:
        content.append(np.load(dir_path + "/" + file))

    return content

def save_model_to_file(model, args, extension=""):
    filename = get_output_path(args, directory="models", filename="m", extension=extension, file_type="pkl")
    print("Saving model to: ", filename)
    pickle.dump(model, open(filename, 'wb'))

def load_model_from_file(model, args, extension=""):
    filename = get_output_path(args, directory="models", filename="m", extension=extension, file_type="pkl")
    print("Loading model from: ", filename)
    return pickle.load(open(filename, 'rb'))

def save_results_to_file(args, results, train_time=None, test_time=None, best_params=None):
    filename = get_output_path(args, filename="results", file_type="txt")
    print("Saving results to: ", filename)
    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write(args.model_name + " - " + args.dataset + "\n\n")

        for key, value in results.items():
            text_file.write("%s: %.5f\n" % (key, value))

        if train_time:
            text_file.write("\nTrain time: %f\n" % train_time)

        if test_time:
            text_file.write("Test time: %f\n" % test_time)

        if best_params:
            text_file.write("\nBest Parameters: %s\n\n\n" % best_params)

def save_hyperparameters_to_file(args, params, results, time=None):
    filename = get_output_path(args, filename="hp_log", file_type="txt")

    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write("Parameters: %s\n\n" % params)

        for key, value in results.items():
            text_file.write("%s: %.5f\n" % (key, value))

        if time:
            text_file.write("\nTrain time: %f\n" % time[0])
            text_file.write("Test time: %f\n" % time[1])

        text_file.write("\n---------------------------------------\n")

def save_results_to_json_file(args, jsondict, resultsname = "results", append=True):
    """ Write the results to a json file. 
        jsondict: A dictionary with results that will be serialized.
        If append=True, the results will be appended to the original file.
        If not, they will be overwritten if the file already exists. 
    """

    filename = get_output_path(args, filename=resultsname, file_type="json", directory="")
    print("Saving scores to: ", filename)
    if append:
        if os.path.exists(filename):
            old_res = json.load(open(filename))
            for k, v in jsondict.items():
                old_res[k].append(v)
        else:
            old_res = {}
            for k, v in jsondict.items():
                old_res[k] = [v]
        jsondict = old_res
    json.dump(jsondict, open(filename, "w"),indent=4)