import sys
import os
import pandas as pd
import warnings
from datetime import datetime
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")

# Set up system path
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import relevant packages
from src.eda import data_info
from src.models import AutoencoderTrainer, AnomalyDetector, AutoencoderTuner

# Config
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

def main():
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="Run Autoencoder hyperparameter tuning.")
    parser.add_argument("--version", type=str, default=None, help="Version identifier for the run")
    
    # Add additional arguments after possibly loading from logs
    parser.add_argument("--metric", type=str, default=None, help="Metric to evaluate (default: auc)")
    parser.add_argument("--n_calls", type=int, default=None, help="Total number of optimization calls for Bayesian tuning")
    parser.add_argument("--random_starts", type=int, default=None, help="Number of initial random points for Bayesian tuning")
    parser.add_argument("--bo_estimator", type=str, default=None, help="Bayesian optimization estimator (default: GP)")
    
    # Add argument for continuing from the last run
    parser.add_argument("--continue_run", type=bool, default=False, help="Continue from the last run")
    
    # Parse complete set of arguments
    args = parser.parse_args()

    log_path = "logs/baseline_tune_log.txt"
    os.makedirs("logs", exist_ok=True)  # Ensure log directory exists

    # --- Resume previous run (if version is specified) ---
    if args.version:
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
            last_args = None
            # Search log for last run with the same version
            version_lines = [line for line in lines if line.startswith("Arguments:") and f"version={args.version}" in line]
            if version_lines:
                parts = version_lines[-1].strip().replace("Arguments: ", "").split(", ")
                last_args = {kv.split("=")[0]: kv.split("=")[1] for kv in parts}
            # Override current args with previously used values
            def get_value(current_val, key, default_val, cast_fn=lambda x: x):
                """Helper to prioritize CLI arg > last run value > default."""
                if args.continue_run:
                    if key == "n_calls" and current_val is not None:    
                        return current_val 
                    elif key in last_args:
                        return cast_fn(last_args[key])
                    else:
                        return default_val          
                else:
                    if current_val:
                        return current_val
                    elif key in last_args:
                        return cast_fn(last_args[key])
                    else:
                        return default_val

            if last_args:
                print("Resuming previous run with version:", args.version)

            args.metric = get_value(args.metric, "metric", "auc")
            args.n_calls = get_value(args.n_calls, "n_calls", 30, int)
            args.random_starts = get_value(args.random_starts, "random_starts", 5, int)
            args.bo_estimator = get_value(args.bo_estimator, "bo_estimator", "GP")
        except Exception as e:
            print(f"Warning: Could not load previous config for version {args.version} due to error: {e}")
    else:
        args.version = datetime.now().strftime("%Y%m%d%H%M")
        args.metric = args.metric if args.metric else "auc"
        args.n_calls = args.n_calls if args.n_calls else 30
        args.random_starts = args.random_starts if args.random_starts else 5
        args.bo_estimator = args.bo_estimator if args.bo_estimator else "GP"

    # --- Start logging this run ---
    start_time = datetime.now()
    with open(log_path, "a") as log_file:
        info_txt = f"\n\n=== Run started at: {start_time} ===\nArguments: version={args.version}"
        info_txt += f", metric={args.metric}, n_calls={args.n_calls}, random_starts={args.random_starts}\n"
        log_file.write(info_txt)
    print(info_txt)
    
    try:
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

        # --- Define hyperparameter grid ---
        param_grid = {
            'hidden_dims': [[64], [64, 32]],
            'batch_size': [64, 128],
            'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4],
            'learning_rate': [0.1, 0.05, 0.02],
            'lam': [1e-4, 1e-3, 1e-2, 1e-1],
            'gamma': [0.001, 0.999],
        }

        # --- Initialize tuner object ---
        tuner = AutoencoderTuner(
            x_train=X_train,
            x_train_val=X_train_validate,
            x_val=X_validate,
            y_val=y_validate,
            real_cols=real_cols,
            binary_cols=binary_cols,
            all_cols=all_cols,
            activation='relu',
            max_epochs=500,
            patience_limit=5,
            version=args.version,
            continue_run=args.continue_run,
            bo_estimator=args.bo_estimator,
        )

        # --- Perform tuning using selected strategy ---
        best_model, best_params, best_score = tuner.bo_tune(
            param_grid, metric=args.metric, n_calls=args.n_calls, random_starts=args.random_starts, eval_num=1
        )

        # --- Save the best hyperparameters ---
        with open("hyperparams/baseline/{}.pkl".format(args.version), "wb") as f:
            pickle.dump(best_params, f)

        # --- Save the best model ---
        best_model.save(f"models/baseline/{args.version}")

        # --- Display final results ---
        print("Best parameters:", best_params)
        print("Performance on the validation set:", best_score)

        # --- Evaluate on the test set ---
        # Read relevant files
        X_test = pd.read_feather("data/processed/X_test.feather")
        y_test = pd.read_feather("data/processed/y_test.feather")

        # After training
        detector = AnomalyDetector(
            model=best_model,
            real_cols=real_cols,
            binary_cols=binary_cols,
            all_cols=all_cols,
            lam=best_params['lam'],
            gamma=best_params['gamma'],
        )

        # Compute scores
        scores = detector._compute_anomaly_scores(X_test)

        # Detect
        y_pred = detector._detect(scores, best_params['threshold'])

        # Evaluate
        metrics = detector._evaluate(y_pred, y_test, scores)
        print("Performance on the test set:", metrics)

        # Save test performance to CSV
        results_path = "results/metrics/baseline.csv"
        metrics["version"] = args.version
        metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if os.path.exists(results_path):
            existing_df = pd.read_csv(results_path)
            existing_df["version"] = existing_df["version"].astype(str)
            metrics_df = pd.concat([existing_df, pd.DataFrame([metrics])], ignore_index=True).sort_values(by=["version", "timestamp"], ascending=True).drop_duplicates(subset=["version"], keep='last')
        else:
            metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(results_path, index=False)

    except Exception as e:
        # --- Log failure with timestamp and error ---
        end_time = datetime.now()
        with open(log_path, "a") as log_file:
            log_file.write(f"Run ended at: {end_time}\n")
            log_file.write(f"Status: FAILED\nError: {str(e)}\n")
        raise  # Re-raise exception after logging

    # --- Final success log ---
    end_time = datetime.now()
    with open(log_path, "a") as log_file:
        log_file.write(f"Run ended at: {end_time}\n")
        log_file.write("Status: SUCCESS\n")

# --- Entry point ---
if __name__ == "__main__":
    main()