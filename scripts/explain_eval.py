# Set the working directory to the parent directory
import sys
import os
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import relevant packages
from src.explainability_test import ShapKernelExplainer, ExplainabilityEvaluator

# Import necessary libraries
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings("ignore")

def main():
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="Run SHAP explanations.")
    parser.add_argument("--metric", type=str, default="F1-Score", help="Tuning objective used to select the best model.")
    parser.add_argument("--model_type", type=str, default="baseline", help="Type of the model to explain.")
    parser.add_argument("--continue_run", type=bool, default=False, help="Continue run if already started.")
    parser.add_argument("--background_method", type=str, default="fixed", help="Background method for SHAP explanations.")
    parser.add_argument("--test", type=bool, default=False)

    # --- Parse arguments ---
    args = parser.parse_args()
    metric = args.metric
    model_type = args.model_type
    continue_run = args.continue_run
    background_method = args.background_method

    # --- Start logging this run ---
    log_path = "logs/explain_eval_log.txt"
    os.makedirs("logs", exist_ok=True)  # Ensure log directory exists
    start_time = datetime.now()
    with open(log_path, "a") as log_file:
        info_txt = f"\n\n=== Run started at: {start_time} ===\nArguments: metric={metric}, model_type={model_type}, continue_run={continue_run}, background_method={background_method}\n"
        log_file.write(info_txt)
    print(info_txt)

    # --- Run SHAP explanations ---
    try:
        eval_init = ExplainabilityEvaluator(model_type=model_type, metric=metric, background_method=background_method, continue_run=continue_run, test=args.test)
        eval_init.evaluate_all_versions()
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

    return print(f"SHAP explanations for {model_type} model with {metric} metric have been generated at {datetime.now()}.")

if __name__ == "__main__":
    main()