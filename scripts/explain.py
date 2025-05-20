# Set the working directory to the parent directory
import sys
import os
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import relevant packages
from src.explainability import ShapKernelExplainer

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
    parser.add_argument("--background_size", type=int, default=100, help="Background size for SHAP explanations.")
    parser.add_argument("--nsamples", type=int, default=500, help="Number of samples for SHAP explanations.")

    # --- Parse arguments ---
    args = parser.parse_args()
    metric = args.metric
    model_type = args.model_type
    continue_run = args.continue_run
    background_size = args.background_size
    nsamples = args.nsamples
    background_method = args.background_method
    print(f"Arguments: metric={metric}, model_type={model_type}, continue_run={continue_run}, background_size={background_size}, nsamples={nsamples}, background_method={background_method}")

    # --- Run SHAP explanations ---
    shap_init = ShapKernelExplainer(model_type=model_type, metric=metric, continue_run=continue_run)
    shap_init.explain_all_versions(background_size=background_size, nsamples=nsamples, background_method=background_method)

    return print(f"SHAP explanations for {model_type} model with {metric} metric have been generated at {datetime.now()}.")

if __name__ == "__main__":
    main()