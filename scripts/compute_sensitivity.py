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
from src.dp_utils import compute_empirical_nsr, compute_T_from_nsr
from src.utils import parse_param_value

def compute_sensitivity(version, x_train, x_train_val, x_val, 
                        real_cols, binary_cols,  all_cols,
                        config, epsilon, delta, noise_mechanism):
    """
    Compute the sensitivity of the model to the features in the dataset.
    
    Parameters:
    - version: str, the version of the model.
    - x_train: pd.DataFrame, the training data.
    - x_train_val: pd.DataFrame, the validation data.
    - real_cols: list, the list of real-valued columns.
    - binary_cols: list, the list of binary columns.
    - all_cols: list, the list of all columns.

    Returns:
    - None
    """

    # Train the equivalent baseline model of the DP-SGD model
    baseline_model = AutoencoderTrainer(
            input_dim=x_train.shape[1],
            real_cols=real_cols,
            binary_cols=binary_cols,
            all_cols=all_cols,
            activation='relu',
            patience_limit=10,
            verbose=False,
            dp_sgd=False,
            post_hoc=False,
            save_tracking=False,
            version=None,
            raise_convergence_error=False,
            **config
        ).train(x_train, x_train_val)
    
    # Evaluate both models
    baseline_scores, noise_multiplier = AnomalyDetector(
        baseline_model, real_cols, binary_cols, all_cols,
        lam=config['lam'], gamma=config['gamma']
    )._compute_anomaly_scores(x_val)

    # Get the scores file of the DP-SGD model
    dpsgd_scores_file_path = f"experiments/scores/dpsgd/{version}.feather"
    dpsgd_scores = pd.read_feather(dpsgd_scores_file_path)['score'].values

    # Compute NSR and derive noise scale
    desired_nsr = compute_empirical_nsr(dpsgd_scores, baseline_scores)

    T = compute_T_from_nsr(
        signal_std=np.std(baseline_scores),
        epsilon=epsilon,
        nsr_target=desired_nsr,
        mechanism=noise_mechanism,
        delta=delta,
    )

    del baseline_model
    del noise_multiplier
    del baseline_scores
    del dpsgd_scores
    del desired_nsr

    return T

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
        sensitivity_df = pd.DataFrame(columns=['version', 'metric', 'epsilon', 'delta', 'noise_mechanism', 'sensitivity', 'compute_time'])

    # --- Check which versions that have not been computed yet or recently updated (compute_time < end_time) ---
    merge_df = pd.merge(dpsgd_versions, sensitivity_df[['version', 'noise_mechanism', 'compute_time', 'sensitivity']], on=['version', 'noise_mechanism'], how='left')
    merge_df = merge_df[merge_df['compute_time'].isnull() | (merge_df['compute_time'] < merge_df['end_time'])]

    # --- Filter out versions that have already been computed ---
    sensitivity_df = merge_df[merge_df['compute_time'].notnull() & (merge_df['compute_time'] >= merge_df['end_time'])][['version', 'tuned_by', 'epsilon', 'delta', 'noise_mechanism', 'sensitivity', 'compute_time']]
    sensitivity_df.rename(columns={'tuned_by': 'metric'}, inplace=True)
    sensitivity_df.to_csv(sensitivity_file_path, index=False)
    
    # --- Compute sensitivity for each model ---
    for i, row in merge_df.iterrows():
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
        T = compute_sensitivity(version, X_train, X_train_validate, X_validate, 
                                real_cols, binary_cols, all_cols, 
                                config, epsilon, delta, noise_mechanism)
        
        print(f"→ Sensitivity: {T}")

        # --- Append the results to the file---
        with open(sensitivity_file_path, 'a') as f:
            f.write(f"{version},{metric},{epsilon},{delta},{noise_mechanism},{T},{pd.Timestamp.now()}\n")

    # --- Print the final message ---
    print("Sensitivity computation completed. Results saved to:", sensitivity_file_path)