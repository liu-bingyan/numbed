import os
import json
import pickle
import numpy as np
from dataclasses import dataclass
# from io_utils import get_output_path

output_dir_flat = "results/"

@dataclass
class ResultFileNameSturct:
    debug = False
    model_name = ""
    dataset = ""
    embedding_structure = ""
    initialization = ""
    filename = ""
    extension = ""

    @classmethod
    def from_filename(cls, filename):
        name_struct = cls()
        filename = filename.split(".")[0]
        components = filename.split("_")

        debug_mode = components[0] == "debug" 
        name_struct.debug = "debug" if debug_mode else ""

        start = 1 if debug_mode else 0
        name_struct.dataset = components[start]
        name_struct.model_name = components[start + 1]
        name_struct.embedding_structure = components[start + 2]
        name_struct.initialization = components[start +3]
        name_struct.filename = components[start + 4]
        name_struct.extension = components[start + 5] 
        return name_struct

    @classmethod
    def from_args(cls, args, filename="", extension=""):
        name_struct = cls()
        name_struct.debug = "debug" if args.small_sample else ""
        name_struct.model_name = args.model_name
        name_struct.dataset = args.dataset
        name_struct.embedding_structure = args.emb_config.split(".")[0].split("_")[1]
        name_struct.initialization = args.emb_config.split(".")[0].split("_")[2]
        name_struct.filename = filename
        name_struct.extension = str(extension)
        return name_struct

    def components(self):
        components = [self.debug] if self.debug else []
        components +=  [self.dataset, self.model_name, self.embedding_structure, 
                self.initialization, self.filename, self.extension]
        return components
                
    def str(self):
        return "_".join(self.components())
    
    def config_init(self):
        other_config = [self.debug] if self.debug else []
        other_config += [self.dataset, self.model_name, self.embedding_structure]
        init = self.initialization
        return "_".join(other_config), init

def save_params_to_file(args, param_dict, filename):
    filename = get_output_path_flat(args, directory="params_pkl", filename=filename, file_type="pkl")
    pickle.dump(param_dict, open(filename, 'wb'))


def load_params_from_file(args, filename):
    filename = get_output_path_flat(args, directory="params_pkl", filename=filename, file_type="pkl")
    return pickle.load(open(filename, 'rb'))


def get_output_path_flat(args, filename, file_type, directory, extension = ""):
 
    dir_path = output_dir_flat

    if directory:
        dir_path = dir_path + "/" + directory

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    full_name = ResultFileNameSturct.from_args(args, filename, extension).str() + "." + file_type
    
    file_path = dir_path + "/" + full_name

    return file_path


def save_scores_to_json_file(args, jsondict, resultsname = "results", append=True):
    """ Write the results to a json file. 
        jsondict: A dictionary with results that will be serialized.
        If append=True, the results will be appended to the original file.
        If not, they will be overwritten if the file already exists. 
    """

    filename = get_output_path_flat(args, filename=resultsname, file_type="json", directory=resultsname)
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

def save_predictions_to_file_flat(arr, args, filename_extension=""):
    filename = get_output_path_flat(args, directory="predictions", filename="predictions", file_type="npy", extension = filename_extension)
    print("Saving predictions to: ", filename)
    np.save(filename, arr)

def save_model_to_file_flat(model, args, filename_extension=""):
    filename = get_output_path_flat(args, directory="models", filename="models", file_type="pkl", extension = filename_extension)
    print("Saving model to: ", filename)
    pickle.dump(model, open(filename, 'wb'))

def load_scores_from_json_file(args, resultsname = "scores"):
    filename = get_output_path_flat(args, filename=resultsname, file_type="json", directory=resultsname)
    print("Loading scores from: ", filename)
    return json.load(open(filename))



# def get_config_filename(args, filename, extension = ""):
#     component_list = []
#     component_list.append("debug" if args.small_data else "")
#     component_list.append(args.dataset)
#     component_list.append(args.model_name)
#     component_list.extend(args.emb_comfig.split(".")[0].split("_")[1:3])
#     component_list.append(filename)
#     if extension != "":
#         component_list.append(str(extension))
#     return "_".join(component_list) 
    # full_name = "debug" if args.small_data else ""
    # full_name +=  "_" + args.dataset
    # # if args.one_hot_encode:
    # #     full_name += "_OneHot"
    # full_name += "_" + args.model_name
    # # full_name += "_BatchSize" + str(args.batch_size) 
    # # if args.scaler != "standard":
    # #     full_name += "_Scaler" + args.scaler.capitalize()

    # full_name += "_" + "_".join(args.emb_comfig.split(".")[0].split("_")[1:3])
    # # if args.numerical_embedding:
    # #     full_name += "_NumEmb"
    # #     full_name += "_Bins" + str(args.num_bins)
    # #     if args.use_M_matrix:
    # #         full_name += "_MMatrix"
    # #     if args.activation != "tanh":
    # #         full_name += "_Act" + args.activation.capitalize()
    # #     if args.initialization != "uniform":
    # #         full_name += "_Init" + args.initialization.capitalize()
    # #      full_name += "_" + args.initialization.capitalize()

# def decode_file_name(filename):
#     components = filename.split("_")
#     args = {}
#     args["small_data"] = "debug" in components
#     args["dataset"] = components[1]
#     args["model"] = components[2]
#     args["embedding_structure"] = components[3]
#     args["initialization"] = components[4]
#     return args