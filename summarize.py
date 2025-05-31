

# for each folder in output_dir, retrive the folder name as model,
    # for each subfolder in the folder, retrive the subfolder name as dataset,
        # for each subsubfolder in the subfolder, retrive the subsubfolder name as run,
            # for each result.json file in the subsubfolder, read scores.json, retrive the timers and scores attribute as a pandas series
        # for each pair of model and dataset, create two dataframes, one for the timers and one for the scores, 
        #   where rows are attributes asscoiated to scores/timers in the json file, and columns are the runs

import os
import json
import pandas as pd
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Summarize the results of the models.")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory containing the output files.")
    parser.add_argument('--report_dir', type=str, default="report", help="Directory to save the report files.")
    return parser

def summarize():
    args = get_parser().parse_args()

    output_dir = args.output_dir
    report_dir = args.report_dir
    os.makedirs(report_dir, exist_ok=True)


    for model in os.listdir(output_dir):
        model_path = os.path.join(output_dir, model)
        if not os.path.isdir(model_path):
            continue

        for dataset in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset)
            if not os.path.isdir(dataset_path):
                continue

            timers_data = {}
            scores_data = {}

            for run in os.listdir(dataset_path):
                
                run_path = os.path.join(dataset_path, run)
                if not os.path.isdir(run_path):
                    continue

                result_file = os.path.join(run_path, "scores.json")
                if not os.path.isfile(result_file):
                    continue

                with open(result_file, "r") as f:
                    result_data = json.load(f)

                timers_data[run] = pd.Series(result_data.get("timers", {}))
                scores_data[run] = pd.Series(result_data.get("scores", {}))
            
            # Create dataframes for timers and scores
            timers_df = pd.DataFrame(timers_data).T.sort_index()
            scores_df = pd.DataFrame(scores_data).T.sort_index()
            # Save the dataframes to CSV files
            timers_csv_path = os.path.join(report_dir, f"{model}_{dataset}_timers.csv")
            timers_df.to_csv(timers_csv_path, index=True, float_format="%.4f")
            scores_csv_path = os.path.join(report_dir, f"{model}_{dataset}_scores.csv")
            scores_df.to_csv(scores_csv_path, index=True, float_format="%.4f")

if __name__ == "__main__":
    summarize()