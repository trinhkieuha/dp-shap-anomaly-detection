# Set the working directory to the parent directory
import sys
import os
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import relevant packages
from src.eda import data_info
from src.models import AutoencoderTrainer, AnomalyDetector, AutoencoderTuner
from src.explainability import ShapKernelExplainer

# Import necessary libraries
import shap
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import pandas as pd

def main():
    dpsgd_versions = pd.read_csv('experiments/perf_summary/dpsgd_val_results.csv')
    dpsgd_versions = dpsgd_versions[dpsgd_versions["tuned_by"]=='AUC'][1:]
    dpsgd_explainer = ShapKernelExplainer(model_type='dpsgd')
    for version in dpsgd_versions["version"].tolist():
        print("Starting Kernel SHAP explanation for version:", version)
        # Explain the model
        shap_values, explainer = dpsgd_explainer.explain(version=str(version), background_size=50)
        # Save the SHAP values
        os.makedirs("results/explainability/dpsgd/", exist_ok=True)
        with open(f"results/explainability/dpsgd/{version}.pkl", "wb") as f:
            pickle.dump(shap_values, f)
        print("Kernel SHAP explanation completed for version:", version)

if __name__ == "__main__":
    main()