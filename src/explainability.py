# Import relevant packages
from src.eda import data_info
from src.models import AutoencoderTrainer, AnomalyDetector, AutoencoderTuner

# Import necessary libraries
import shap
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
import re

class ShapKernelExplainer:
    """
    Class to handle SHAP KernelExplainer for anomaly detection models.
    """

    def __init__(self, model_type='baseline'):
        """
        Initializes the ShapKernelExplainer with model type and loads test data.

        Parameters: 
        - model_type: str, type of the model to be used (default is 'baseline')
        """
        # Set the model type
        self.model_type = model_type

        # Read relevant files
        self.X_test = pd.read_feather("data/processed/X_test.feather")
        self.y_test = pd.read_feather("data/processed/y_test.feather")

        # Extract variable types from metadata
        self.var_info = data_info(self.X_test)
        self.all_cols = self.X_test.columns
        self.real_cols = self.var_info[self.var_info["var_type"] == "numerical"]["var_name"].tolist()
        self.binary_cols = self.var_info[self.var_info["var_type"] == "binary"]["var_name"].tolist()
        
        # Get the model information
        self.model_info = pd.read_csv(f"experiments/perf_summary/{model_type}_val_results.csv")
        self.model_info["version"] = self.model_info["version"].astype(str)

        # Filter the model information based on the model type
        if self.model_info.empty:
            raise ValueError(f"No model information found for model type {model_type}.")

    def _version_info_extract(self, version=None):


        if version is None:
            raise ValueError("Version must be provided for model extraction.")

        # Load the model and parameters
        model = tf.keras.models.load_model(f"models/{self.model_type}/{version}")

        # Check if the version is provided
        version_info = self.model_info[self.model_info["version"] == version]
        if version_info.empty:
            raise ValueError(f"No model information found for version {version}.")
        
        # Extract parameters
        lam = version_info["lam"].values[0]
        gamma = version_info["gamma"].values[0]

        # Extract noise mechanism if applicable
        noise_multiplier = None
        if self.model_type == "posthoc_dp":
            noise_mechanism = version_info["noise_mechanism"].values[0]
            # Extract noise multiplier
            for file in os.listdir(f"experiments/scores/posthoc_dp/"):
                if f"{version}_noise" in file and file.endswith(".feather"):
                    noise_multiplier = float(re.search(r'noise(\d+(\.\d+)?)', file).group(1))
                    break
            if noise_multiplier is None:
                raise ValueError(f"Noise multiplier not found for version {version}")
        else:
            noise_mechanism = None
        
        # Extract epsilon and delta if applicable
        if self.model_type in ["posthoc_dp", "dpsgd"]:
            epsilon = version_info["epsilon"].values[0]
            delta = version_info["delta"].values[0]
        else:
            epsilon = None
            delta = None

        # Initialize the anomaly detector
        detector = AnomalyDetector(
            model=model,
            real_cols=self.real_cols,
            binary_cols=self.binary_cols,
            all_cols=self.all_cols,
            lam=lam,
            gamma=gamma,
            post_hoc=(self.model_type == "posthoc_dp"),
            noise_mechanism=noise_mechanism,
            target_epsilon=epsilon,
            delta=delta
        )

        return detector, noise_multiplier

    def _explain_with_kernelshap(self, detector, background_size=100, noise_multiplier=None):
        """
        Applies SHAP KernelExplainer to anomaly scores from a detector.

        Parameters:
        - detector: AnomalyDetector object, the trained anomaly detection model
        - background_size: int, number of samples to use as background

        Returns:
        - shap_values: list of arrays, SHAP values for each test instance
        - explainer: the trained SHAP explainer object
        """
        # Use a subset of the test set as background (mean + diverse set is typical)
        background = self.X_test.sample(n=min(background_size, len(self.X_test)), random_state=42)

        # Define a wrapper function for SHAP to call
        def anomaly_scorer(X_input):
            X_df = pd.DataFrame(X_input, columns=self.X_test.columns)
            return detector._compute_anomaly_scores(X_df, noise_multiplier=noise_multiplier)

        # Instantiate the KernelExplainer
        explainer = shap.KernelExplainer(anomaly_scorer, background)

        # Compute SHAP values for all test instances
        shap_values = explainer.shap_values(self.X_test, nsamples=100)

        return shap_values, explainer
    
    def explain(self, version=None, background_size=100):
        """
        Main method to explain the model using SHAP KernelExplainer.

        Parameters:
        - version: str, version of the model to be explained
        - background_size: int, number of samples to use as background

        Returns:
        - shap_values: list of arrays, SHAP values for each test instance
        - explainer: the trained SHAP explainer object
        """
        # Extract version information and load the model
        detector, noise_multiplier = self._version_info_extract(version)

        # Apply SHAP KernelExplainer
        shap_values, explainer = self._explain_with_kernelshap(detector, background_size, noise_multiplier)

        return shap_values, explainer