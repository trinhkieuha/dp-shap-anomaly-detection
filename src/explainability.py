# Import relevant packages
from src.eda import data_info
from src.models import AnomalyDetector

# Import necessary libraries
import shap
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
import re
from tqdm import tqdm

class ShapKernelExplainer:
    """
    Class to handle SHAP KernelExplainer for anomaly detection models.
    """

    def __init__(self, model_type='baseline', metric='F1-Score'):
        """
        Initializes the ShapKernelExplainer with model type and loads test data.

        Parameters: 
        - model_type: str, type of the model to be used (default is 'baseline')
        """
        # Set the model type
        self.model_type = model_type

        # Read relevant files
        self.X_train = pd.read_feather("data/processed/X_train.feather")
        self.X_test = pd.read_feather("data/processed/X_test.feather")
        self.y_test = pd.read_feather("data/processed/y_test.feather")

        # Extract variable types from metadata
        self.var_info = data_info(self.X_test)
        self.all_cols = self.X_test.columns
        self.real_cols = self.var_info[self.var_info["var_type"] == "numerical"]["var_name"].tolist()
        self.binary_cols = self.var_info[self.var_info["var_type"] == "binary"]["var_name"].tolist()

        # Extract the index of the features that are real-valued and binary
        col_to_idx = {col: i for i, col in enumerate(self.all_cols)}
        self.real_idx = [col_to_idx[c] for c in self.real_cols]
        self.binary_idx = [col_to_idx[c] for c in self.binary_cols]

        # Get the model information
        self.model_info = pd.read_csv(f"experiments/perf_summary/{model_type}_val_results.csv")
        self.model_info = self.model_info[self.model_info["tuned_by"] == metric]
        self.model_info["version"] = self.model_info["version"].astype(str)

        # Filter the model information based on the model type
        if self.model_info.empty:
            raise ValueError(f"No model information found for model type {model_type}.")

    def _version_info_extract(self, version=None):
        """
        Extracts version information and loads the model.
        Parameters:
        - version: str, version of the model to be loaded
        Returns:
        - detector: AnomalyDetector object, the loaded anomaly detection model
        - noise_multiplier: float, noise multiplier if applicable
        """
        # Check if version is provided
        if version is None:
            raise ValueError("Version must be provided for model extraction.")

        # Load the model and parameters
        model = tf.keras.models.load_model(f"models/{self.model_type}/{version}")

        # Check if the version is provided
        version_info = self.model_info[self.model_info["version"] == version]
        if version_info.empty:
            raise ValueError(f"No model information found for version {version}.")
        
        # Load the parameters
        with open(f"hyperparams/{self.model_type}/{version}.pkl", "rb") as f:
            params = pickle.load(f)

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
            lam=params["lam"],
            gamma=params["gamma"],
            post_hoc=(self.model_type == "posthoc_dp"),
            noise_mechanism=noise_mechanism,
            target_epsilon=epsilon,
            delta=delta
        )

        return model, params

    def _sample_background_set(self, x_anomaly, num_background=100):
        """
        Samples a background set for SHAP KernelExplainer using influence weights.
        Parameters:
        - x_anomaly: array-like, the instance to explain
        - num_background: int, number of samples to use as background
        Returns:
        - background_set: array-like, sampled background set
        - influence_weights: array-like, influence weights for the background set
        """
        # Ensure x_anomaly is 2D and X_train is an array
        x_anomaly = np.atleast_2d(x_anomaly)  # ensure shape (1, d)
        X_train = np.asarray(self.X_train)    # ensure array

        # Compute squared L2 distances
        dists_squared = np.sum((X_train - x_anomaly) ** 2, axis=1)

        # Compute RBF-based influence weights
        dists = np.linalg.norm(X_train - x_anomaly, axis=1)
        sigma = np.median(dists)
        rho = np.exp(-dists_squared / sigma**2)
        influence_weights = rho / np.sum(rho)

        # Sample with replacement using influence weights
        indices = np.random.choice(len(X_train), size=num_background, p=influence_weights, replace=True)
        background_set = X_train[indices]

        return background_set, influence_weights
    
    def _compute_reconstruction_error(self, x_anomaly, x_recon, params):
        eps = 1e-7
        errors = np.zeros_like(x_anomaly)  # same shape as x_anomaly

        # Compute MSE for real-valued features
        if self.real_idx:
            x_r = x_anomaly[:, self.real_idx]
            xhat_r = x_recon[:, self.real_idx]
            errors[:, self.real_idx] = (x_r - xhat_r) ** 2 * params["gamma"]

        # Compute cross-entropy for binary features
        if self.binary_idx:
            x_b = x_anomaly[:, self.binary_idx]
            xhat_b = x_recon[:, self.binary_idx]
        errors[:, self.binary_idx] = -x_b * np.log(xhat_b + eps) - (1 - x_b) * np.log(1 - xhat_b + eps)

        return errors

    def _explain_with_kernelshap(self, x_anomaly, model, params, background_size=100, noise_multiplier=None):
        """
        Applies SHAP KernelExplainer to anomaly scores from a detector.

        Parameters:
        - x_anomaly: array-like, the instance to explain
        - model: AnomalyDetector object, the anomaly detection model
        - params: dict, parameters for the model
        - background_size: int, number of samples to use as background
        - noise_multiplier: float, noise multiplier if applicable
        Returns:
        - shap_values: list of arrays, SHAP values for each test instance
        - explainer: the trained SHAP explainer object
        """
        # Ensure x_anomaly is 2D and X_train is an array
        x_anomaly = np.atleast_2d(x_anomaly)  # ensure shape (1, d)

        # Compute reconstruction and per-feature error
        x_tensor = tf.convert_to_tensor(x_anomaly)
        x_recon_tensor = model(x_tensor, training=False)
        x_recon = x_recon_tensor.numpy()

        # Compute the total error
        errors = self._compute_reconstruction_error(x_anomaly, x_recon, params).flatten()
        total_error = np.sum(errors)
        sorted_idx = np.argsort(errors)[::-1]
        cumsum = np.cumsum(errors[sorted_idx])
        m = np.searchsorted(cumsum, 0.85 * total_error) + 1
        top_features = sorted_idx[:m]

        # Sample background set
        background_set, influence_weights = self._sample_background_set(x_anomaly, num_background=background_size)

        # Define local prediction wrapper using TensorFlow model
        def predict_fn(X):
            X_tensor = tf.convert_to_tensor(X.astype(np.float32))
            X_recon = model(X_tensor, training=False).numpy()
            errors = self._compute_reconstruction_error(X, X_recon, params)
            return np.sum(errors[:, top_features], axis=1)

        # Fit SHAP KernelExplainer
        explainer = shap.KernelExplainer(model=predict_fn, data=background_set)
        shap_values = explainer.shap_values(x_anomaly, silent=True)

        return shap_values
    
    def _explain(self, version=None, background_size=100):
        """
        Main method to explain the model using SHAP KernelExplainer.

        Parameters:
        - version: str, version of the model to be explained
        - background_size: int, number of samples to use as background

        Returns:
        - None, but saves the SHAP values to a CSV file
        """
        # Get the anomaly predictions and the corresponding data
        y_pred = pd.read_feather(f"experiments/predictions/{self.model_type}/{version}_pred.feather")
        X_anomaly = self.X_test[y_pred['anomaly']==1].values

        # Initialize the file to save the results
        os.makedirs(f"results/explainability/{self.model_type}", exist_ok=True)
        with open(f"results/explainability/{self.model_type}/{version}.csv", "w") as f:
            f.write(",".join(self.all_cols))
            f.write("\n")

        # Load the model and parameters
        model, params = self._version_info_extract(version=version)

        # Loop through each anomaly and compute SHAP values
        for x_anomaly in tqdm(X_anomaly, desc=f"Explaining anomalies for {self.model_type} model version: {version}"):
            # Compute the SHAP values for the current anomaly
            shap_values = self._explain_with_kernelshap(x_anomaly, model, params, background_size=background_size)
            
            # Save the SHAP values to a CSV file
            with open(f"results/explainability/{self.model_type}/{version}.csv", "a") as f:
                f.write(",".join([str(val) for val in shap_values.reshape(-1)]))
                f.write("\n")

    def explain_all_versions(self, background_size=100):
        """
        Explains all versions of the model using SHAP KernelExplainer.

        Parameters:
        - background_size: int, number of samples to use as background

        Returns:
        - None, but saves the SHAP values for each version to a CSV file
        """
        # Loop through each version and explain
        for version in self.model_info["version"].unique():
            self._explain(version=version, background_size=background_size)