import sys
import os
import pandas as pd
import warnings
from datetime import datetime
import argparse
import pickle
import math
import warnings
warnings.filterwarnings("ignore")

# Set up system path
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import relevant packages
from src.eda import data_info
from src.models import AnomalyDetector, AutoencoderTuner

# Config
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

def main():
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="Run DP-SGD Autoencoder hyperparameter tuning.")
    parser.add_argument("--version", type=str, default=None, help="Version identifier for the run")
    parser.add_argument("--tune_method", type=str, choices=["sequential", "bayesian", None], default=None,
                        help="Tuning method to use: 'sequential' or 'bayesian' (default: sequential)")

    # Add additional arguments after possibly loading from logs
    parser.add_argument("--metric", type=str, default=None, help="Metric to evaluate (default: auc)")
    parser.add_argument("--random_size", type=int, default=None, help="Number of random samples (default: 5)")
    parser.add_argument("--n_calls", type=int, default=None, help="Total number of optimization calls for Bayesian tuning")
    parser.add_argument("--random_starts", type=int, default=None, help="Number of initial random points for Bayesian tuning")

    # Add arguments for differential privacy
    parser.add_argument("--epsilon", type=str, default=None, help="Epsilon value for differential privacy")
    parser.add_argument("--delta", type=str, default=None, help="Delta value for differential privacy")

    # Add argument for continuing from the last run
    parser.add_argument("--continue_run", type=bool, default=False, help="Continue from the last run")
    
    # Parse complete set of arguments
    args = parser.parse_args()

    log_path = "logs/dpsgd_tune_log.txt"
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

            if last_args:
                print("Resuming previous run with version:", args.version)

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

            args.tune_method = get_value(args.tune_method, "tune_method", "bayesian")
            args.metric = get_value(args.metric, "metric", "auc")
            args.random_size = get_value(args.random_size, "random_size", 5, int)
            args.n_calls = get_value(args.n_calls, "n_calls", 30, int)
            args.random_starts = get_value(args.random_starts, "random_starts", 5, int)
            args.epsilon = get_value(args.epsilon, "epsilon", "1")
            print(f"Using epsilon: {args.epsilon}")
            args.delta = get_value(args.delta, "delta", "1e-5")
        except Exception as e:
            print(f"Warning: Could not load previous config for version {args.version} due to error: {e}")
    else:
        args.version = datetime.now().strftime("%Y%m%d%H%M")
        args.tune_method = args.tune_method if args.tune_method else "bayesian"
        args.metric = args.metric if args.metric else "auc"
        args.random_size = args.random_size if args.random_size else 5
        args.n_calls = args.n_calls if args.n_calls else 30
        args.random_starts = args.random_starts if args.random_starts else 5
        args.epsilon = args.epsilon if args.epsilon else "1"
        args.delta = args.delta if args.delta else "1e-5"

    # --- Start logging this run ---
    start_time = datetime.now()
    with open(log_path, "a") as log_file:
        info_txt = f"\n\n=== Run started at: {start_time} ===\nArguments: version={args.version}, tune_method={args.tune_method}, epsilon={args.epsilon}, delta={args.delta}"
        if args.version == 'sequential':
            info_txt += f", metric={args.metric}, random_size={args.random_size}\n"
        else:
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
        base_grid = [0.001, 0.00005]
        scaled_grid = [round(l * float(args.epsilon) ** 1.5, 3) for l in base_grid]
        learning_rate_grid = [scaled_grid[0] + 0.00001, max(scaled_grid[1] - 0.00001, 1e-6)]
        param_grid = {
            'hidden_dims': [[64], [64, 32]],
            'batch_size': [64, 128, 150],
            'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4],
            'learning_rate': learning_rate_grid,
            'lam': [1e-4, 1e-3, 1e-2, 1e-1],
            'gamma': [0.001, 0.999],
            'l2norm_pct': [75, 95],
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
            patience_limit=10,
            version=args.version,
            dp_sgd=True,
            target_epsilon=float(args.epsilon),
            delta=float(args.delta),
            continue_run=args.continue_run,
        )

        # --- Perform tuning using selected strategy ---
        if args.tune_method == "sequential":
            best_model, best_params, best_score = tuner.sequential_tune(
                param_grid, metric=args.metric, random_size=args.random_size
            )
        else:
            best_model, best_params, best_score = tuner.bo_tune(
                param_grid, metric=args.metric, n_calls=args.n_calls, random_starts=args.random_starts, eval_num=int(max(1, 6-float(args.epsilon)))
            )

        # --- Save the best hyperparameters ---
        os.makedirs("hyperparams/dpsgd", exist_ok=True)  # Ensure hyperparams/dpsgd directory exists
        with open("hyperparams/dpsgd/{}.pkl".format(args.version), "wb") as f:
            pickle.dump(best_params, f)

        # --- Save the best model ---
        os.makedirs("models/dpsgd", exist_ok=True)  # Ensure models/dpsgd directory exists
        best_model.save(f"models/dpsgd/{args.version}")

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
        results_path = f"results/metrics/dpsgd.csv"
        metrics["version"] = args.version
        metrics["eps"] = args.epsilon
        metrics["delta"] = args.delta
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