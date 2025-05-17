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

    # --- Parse arguments ---
    args = parser.parse_args()
    metric = args.metric
    model_type = args.model_type

    # --- Run SHAP explanations ---
    shap_init = ShapKernelExplainer(model_type=model_type, metric=metric)
    shap_init.explain_all_versions()

    return print(f"SHAP explanations for {model_type} model with {metric} metric have been generated at {datetime.now()}.")

if __name__ == "__main__":
    main()