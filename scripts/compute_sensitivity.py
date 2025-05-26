import pandas as pd
import sys
import os
import numpy as np
import ast
import warnings
warnings.filterwarnings("ignore")

# Set up system path
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import relevant packages
from src.eda import data_info
from src.models import AnomalyDetector, AutoencoderTrainer
from src.dp_utils import max_per_sample_sensitivity

if __name__ == "__main__":

    # --- Load data ---
    X_train = pd.read_feather("data/processed/X_train.feather")
    X_train_validate = pd.read_feather("data/processed/X_train_validate.feather")
    X_validate = pd.read_feather("data/processed/X_validate.feather")
    y_validate = pd.read_feather("data/processed/y_validate.feather")

    # --- Extract variable types from metadata ---
    var_info = data_info(X_train)
    all_cols = X_train.columns
    real_cols = var_info[var_info["var_type"] == "numerical"]["var_name"].tolist()
    binary_cols = var_info[var_info["var_type"] == "binary"]["var_name"].tolist()

    # --- Get the model version and parameters ---
    dpsgd_versions = pd.read_csv("experiments/perf_summary/dpsgd_val_results.csv")
    version_list = dpsgd_versions['version'].tolist()
    version_noise_mechanism = pd.merge(pd.DataFrame({'version': version_list}), pd.DataFrame({'noise_mechanism': ['gaussian', 'laplace']}), how='cross')
    dpsgd_versions = pd.merge(dpsgd_versions, version_noise_mechanism, on='version', how='left')

    # --- Read the current sensitivity file ---
    sensitivity_file_path = "experiments/perf_summary/sensitivity.csv"
    if os.path.exists(sensitivity_file_path):
        print("Loading existing sensitivity file...")
        sensitivity_df = pd.read_csv(sensitivity_file_path)
    else:
        # Create an empty file if the file does not exist using with open
        print("Creating new sensitivity file...")
        sensitivity_df = pd.DataFrame(columns=['version', 'metric', 'epsilon', 'delta', 'noise_mechanism', 'sensitivity', 'compute_time'])

    # --- Check which versions that have not been computed yet or recently updated (compute_time < end_time) ---
    merge_df = pd.merge(dpsgd_versions, sensitivity_df[['version', 'noise_mechanism', 'compute_time', 'sensitivity']], on=['version', 'noise_mechanism'], how='left')
    new_versions = merge_df[merge_df['compute_time'].isnull() | (merge_df['compute_time'] < merge_df['end_time'])]

    # --- Filter out versions that have already been computed ---
    sensitivity_df = merge_df[merge_df['compute_time'].notnull() & (merge_df['compute_time'] >= merge_df['end_time'])][['version', 'tuned_by', 'epsilon', 'delta', 'noise_mechanism', 'sensitivity', 'compute_time']]
    sensitivity_df.rename(columns={'tuned_by': 'metric'}, inplace=True)
    sensitivity_df['metric'] = sensitivity_df['metric'].str.lower().str.replace('-', '_')
    sensitivity_df.to_csv(sensitivity_file_path, index=False)
    
    # --- Compute sensitivity for each model ---
    for i, row in new_versions.iterrows():
        version = row['version']
        metric = row['tuned_by'].lower().replace('-', '_')
        epsilon = row['epsilon']
        delta = row['delta']
        noise_mechanism = row['noise_mechanism']
        
        # --- Load the model parameters ---
        config = {
            'hidden_dims': tuple(ast.literal_eval(str(row['hidden_dims']))),
            'dropout_rate': float(row['dropout_rate']),
            'lam': float(row['lam']),
            'gamma': float(row['gamma']),
            'batch_size': int(row['batch_size']),
            'learning_rate': float(row['learning_rate']),
            'max_epochs': int(row['max_epochs']),
        }

        # --- Compute sensitivity ---
        print(f"Computing sensitivity for version {version} with noise mechanism {noise_mechanism}")
        T = max_per_sample_sensitivity(len(real_cols), len(binary_cols), config['gamma'])

        print(f"→ Sensitivity: {T}")

        # --- Append the results to the file---
        with open(sensitivity_file_path, 'a') as f:
            f.write(f"{version},{metric},{epsilon},{delta},{noise_mechanism},{T},{pd.Timestamp.now()}\n")

    # --- Print the final message ---
    print("Sensitivity computation completed. Results saved to:", sensitivity_file_path)