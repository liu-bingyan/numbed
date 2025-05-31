#!/usr/bin/env python3
import os
import sys
import pandas as pd
import argparse
import re
from pathlib import Path
import glob
import json
import numpy as np

def find_metric_columns(df, regression_metric, classification_metric):
    """
    Find columns related to either regression or classification metrics
    Returns the mean and std column names if found, and the metric type used
    """
    # Define possible column name patterns for regression metrics
    reg_patterns = [f'{regression_metric} - mean', f'{regression_metric}-mean', 
                   f'{regression_metric}_mean', f'{regression_metric} mean']
    reg_std_patterns = [f'{regression_metric} - std', f'{regression_metric}-std', 
                       f'{regression_metric}_std', f'{regression_metric} std']
    
    # Define possible column name patterns for classification metrics
    cls_patterns = [f'{classification_metric} - mean', f'{classification_metric}-mean', 
                   f'{classification_metric}_mean', f'{classification_metric} mean']
    cls_std_patterns = [f'{classification_metric} - std', f'{classification_metric}-std', 
                       f'{classification_metric}_std', f'{classification_metric} std']
    
    # Check for regression metric columns
    reg_mean_col = None
    for pattern in reg_patterns:
        matching_cols = [col for col in df.columns if pattern in col]
        if matching_cols:
            reg_mean_col = matching_cols[0]
            break
    
    reg_std_col = None
    for pattern in reg_std_patterns:
        matching_cols = [col for col in df.columns if pattern in col]
        if matching_cols:
            reg_std_col = matching_cols[0]
            break
    
    # Check for classification metric columns
    cls_mean_col = None
    for pattern in cls_patterns:
        matching_cols = [col for col in df.columns if pattern in col]
        if matching_cols:
            cls_mean_col = matching_cols[0]
            break
    
    cls_std_col = None
    for pattern in cls_std_patterns:
        matching_cols = [col for col in df.columns if pattern in col]
        if matching_cols:
            cls_std_col = matching_cols[0]
            break
    
    # Determine which metric type is present
    if reg_mean_col and reg_std_col:
        return reg_mean_col, reg_std_col, regression_metric
    elif cls_mean_col and cls_std_col:
        return cls_mean_col, cls_std_col, classification_metric
    else:
        return None, None, None

def extract_model_name(filename):
    """Extract model name from filename (first token before underscore)"""
    basename = os.path.basename(filename)
    match = re.match(r'^([^_]+)', basename)
    if match:
        return match.group(1)
    return None

def extract_embedding_dim(runname):
    if runname == 'default':
        return 0
    
    splited = runname.split('_')
    emb_dim = runname.split('_')[1]
    emb_dim =''.join(c for c in emb_dim if c.isdigit())

    return emb_dim
def remove_embedding_dim(runname):
    if runname == 'default':
        return runname
    
    splited = runname.split('_')
    splited[1] = ''.join(c for c in splited[1] if not c.isdigit())
    runname = '_'.join(splited[:])
    return runname

def simplify_run_name(runname):
    #print(runname)
    if runname == 'default':
        return runname
    dim = extract_embedding_dim(runname)
    splited = runname.split('_')
    #print(splited)
    variant = 'p'
    for i in range(len(splited)):
        if splited[i] == 'ctrl':
            splited[i] = 'F'
        elif splited[i] == 'emb':
            splited[i] = 'L'
        elif splited[i] == 'tanh' or splited[i] == 'relu':
            splited[i] = splited[i][0].upper()
        elif splited[i] == 'M':
            splited[i] = 'L'
        elif splited[i] == 'alpha' or splited[i] == 'standard' or splited[i] == 'rtdl':
            splited[i] = splited[i][0]
        elif splited[i] == 'lr' or splited[i] == 'tlr' or splited[i] == 'lrlr' or splited[i] == 'tlrlr':
            variant = splited[i][0]
            splited[i] = ""
        else:
            splited[i] = ""
    trimmed = [s for s in splited if s]
    if trimmed[-1] not in ['s', 'a', 'r']:
        splited = [variant] + ["-"] + trimmed 
    else:
        splited = [variant] + ["-"] + trimmed[:-1]+ ["-"] + trimmed[-1:]
    simplified_name = ''.join(splited)
    #print(splited,dim)
    #print(simplified_name)
    return simplified_name
    

def prepare_report_for_aggregation(df, mean_col, std_col, metric_type, dataset_name):

    
    DN = dataset_name[:2].upper()
    col_idx = 'embedding-model'
    col_mean = f'{DN}-{metric_type}-mean'
    col_std = f'{DN}-{metric_type}-std'
    col_dim =f'{DN}-dim'
    new_dataset_result = df[[mean_col, std_col]].reset_index()
    new_dataset_result.columns = [col_idx,col_mean, col_std]
    new_dataset_result[col_dim] = new_dataset_result[col_idx].apply(extract_embedding_dim)
    new_dataset_result[col_idx] = new_dataset_result[col_idx].apply(remove_embedding_dim)
    
    
    simplified_result = new_dataset_result[[col_idx,col_mean]].copy()
    simplified_result[col_idx] = simplified_result[col_idx].apply(simplify_run_name)
    simplified_result.set_index(col_idx, inplace=True)
    simplified_result.columns = [col_mean.strip('-mean')]

    new_dataset_result = new_dataset_result[[col_idx, col_dim, col_mean, col_std]]
    new_dataset_result.set_index(col_idx, inplace=True)
    return new_dataset_result,simplified_result
    
    
    
def aggregate_reports(report_dir, regression_metric='MSE', classification_metric='AUC'):
    """
    Aggregate reports from CSV files in the report directory
    
    Args:
        report_dir: Directory containing CSV report files
        regression_metric: Metric to use for regression tasks (default: MSE)
        classification_metric: Metric to use for classification tasks (default: AUC)
    """
    # Create aggregated directory if it doesn't exist
    report_path = Path(report_dir)
    aggregated_dir = report_path / 'aggregated'
    simplified_dir = report_path / 'simplified'
    aggregated_dir.mkdir(exist_ok=True)
    simplified_dir.mkdir(exist_ok=True)
    
    # Find all CSV files in the report directory
    csv_files = list(report_path.glob('*.csv'))
    
    # Group files by model name
    model_files = {}
    for file in csv_files:
        model_name = extract_model_name(file)
        if model_name:
            if model_name not in model_files:
                model_files[model_name] = []
            model_files[model_name].append(file)
    
    # Process each model's files
    for model_name, files in model_files.items():
        print(f"Processing {model_name}...")
        
        # Initialize an empty DataFrame to store aggregated results
        aggregated_data = pd.DataFrame()
        simplified_data = pd.DataFrame()
        aggregated_timer_data = pd.DataFrame()
        simplified_timer_data = pd.DataFrame()
        
        # Process each file for this model
        for file in files:
            if file.name.endswith('scores.csv'):
                df = pd.read_csv(file, index_col=0)
                
                # Find the metric columns
                mean_col, std_col, metric_type = find_metric_columns(
                    df, regression_metric, classification_metric)
                
                if mean_col and std_col:
                    # Extract dataset name from filename
                    dataset_name = file.stem.split('_', 1)[1].split('_')[0]
                    
                    # Create a new row with dataset name and metrics
                    new_dataset_result,simplified_result = prepare_report_for_aggregation(df, mean_col, std_col, metric_type, dataset_name)
                    # Append to aggregated data
                    aggregated_data = pd.concat([aggregated_data, new_dataset_result], axis=1)
                    simplified_data = pd.concat([simplified_data, simplified_result], axis=1)
                else:
                    print(f"  Warning: Could not find either {regression_metric} or {classification_metric} columns in {file}")
            
            # Process timers.csv files
            if file.name.endswith('timers.csv'):
                # Read the CSV file
                dataset_name = file.stem.split('_', 1)[1].split('_')[0]
                timer_df = pd.read_csv(file,index_col=0)
                idx = pd.Series(timer_df.index)
                idx = idx.apply(remove_embedding_dim)
                idx = idx.apply(simplify_run_name)
                timer_df.index = idx
                epochs = timer_df['average_epochs']               
                tpe = timer_df['time_per_epoch']
                new_timer_result = pd.DataFrame({f'{dataset_name}-epochs':epochs, 
                                                 f'{dataset_name}-tpe':tpe})
                new_simplified_timer_result = pd.DataFrame({f'{dataset_name}-epochs':epochs})
                aggregated_timer_data = pd.concat([aggregated_timer_data, new_timer_result], axis=1)
                simplified_timer_data = pd.concat([simplified_timer_data, new_simplified_timer_result], axis=1)
                

                
                    
        
        # Custom sorting function for 3-nested partial order
        def custom_sort_key(idx):
            idx_str = str(idx)
            # First level: 'default' comes first
            if idx_str == 'default':
                return (0, 0, 0, 0, idx_str)
            # Second level: rows starting with 'l' come next
            elif idx_str.startswith('l'):
                # Get the string without the last character for lexicographical comparison
                prefix = idx_str[:-1] if len(idx_str) > 1 else ''
                # Get the last character for 's' vs 'a' comparison
                last_char = idx_str[-1] if idx_str else ''
                # Priority for last character: 's' = 0, 'a' = 1
                last_char_priority = 0 if last_char == 's' else (1 if last_char == 'a' else 2)
                return (1, len(idx_str), prefix, last_char_priority, idx_str)
            elif idx_str.startswith('t'):
                # Get the string without the last character for lexicographical comparison
                prefix = idx_str[:-1] if len(idx_str) > 1 else ''
                # Get the last character for 's' vs 'a' comparison
                last_char = idx_str[-1] if idx_str else ''
                # Priority for last character: 's' = 0, 'a' = 1
                last_char_priority = 0 if last_char == 's' else (1 if last_char == 'a' else 2)
                return (2, len(idx_str), prefix, last_char_priority, idx_str)
            # Third level: sort by length (longer later)
            else:
                # Get the string without the last character for lexicographical comparison
                prefix = idx_str[:-1] if len(idx_str) > 1 else ''
                # Get the last character for 's' vs 'a' comparison
                last_char = idx_str[-1] if idx_str else ''
                # Priority for last character: 's' = 0, 'a' = 1
                last_char_priority = 0 if last_char == 's' else (1 if last_char == 'a' else 2)
                return (3, len(idx_str), prefix, last_char_priority, idx_str)
        
        # Sort by custom key
        if not aggregated_data.empty:
            # Sort the DataFrame based on the custom key
            sorted_indices = sorted(aggregated_data.index, key=custom_sort_key)
            aggregated_data = aggregated_data.reindex(sorted_indices)
        
        if not simplified_data.empty:
            # Sort the DataFrame based on the custom key
            sorted_indices = sorted(simplified_data.index, key=custom_sort_key)
            simplified_data = simplified_data.reindex(sorted_indices)
        
        # Save the aggregated data
        if not aggregated_data.empty:
            output_file = os.path.join(aggregated_dir, f"{model_name}.csv")
            aggregated_data.to_csv(output_file)
            print(f"  Saved aggregated data to {output_file}")

            output_file = os.path.join(simplified_dir, f"{model_name}.csv")
            simplified_data.to_csv(output_file)
            print(f"  Saved simplified data to {output_file}")
            
        # Save the timer data
        if not aggregated_timer_data.empty:
            aggregate_timer_csv_path = aggregated_dir / f"{model_name}_time.csv"
            simplified_timer_csv_path = simplified_dir / f"{model_name}_time.csv"
            aggregated_timer_data.to_csv(aggregate_timer_csv_path, float_format="%.4f")
            simplified_timer_data.to_csv(simplified_timer_csv_path, float_format="%.4f")
            print(f"  Saved aggregated timer data to {aggregate_timer_csv_path}")
            print(f"  Saved simplified timer data to {simplified_timer_csv_path}")
        else:
            print(f"  No data to aggregate for {model_name}")




def main():
    parser = argparse.ArgumentParser(description='Aggregate CSV reports by model')
    parser.add_argument('--report_dir', required=True, help='Directory containing CSV report files')
    parser.add_argument('--regression_metric', default='MSE', 
                        help='Metric to use for regression tasks (default: MSE)')
    parser.add_argument('--classification_metric', default='AUC', 
                        help='Metric to use for classification tasks (default: AUC)')
    
    args = parser.parse_args()
    
    # Check if report directory exists
    if not os.path.isdir(args.report_dir):
        print(f"Error: Report directory '{args.report_dir}' does not exist")
        sys.exit(1)
    
    # Aggregate reports
    aggregate_reports(args.report_dir, args.regression_metric, args.classification_metric)
    print("Aggregation complete!")

if __name__ == "__main__":
    main()
