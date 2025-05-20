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
import gc
from sklearn.preprocessing import MinMaxScaler

class ShapKernelExplainer:
    """
    Class to handle SHAP KernelExplainer for anomaly detection models.
    """

    def __init__(self, model_type='baseline', metric='F1-Score', continue_run=False):
        """
        Initializes the ShapKernelExplainer with model type and loads test data.

        Parameters: 
        - model_type: str, type of the model to be used (default is 'baseline')
        - metric: str, tuning objective used to select the best model (default is 'F1-Score')
        - continue_run: bool, whether to continue a previous run (default is False)
        """
        # Set the model type
        self.model_type = model_type
        self.continue_run = continue_run

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

        return model, params

    def _sample_iw_background_set(self, x_anomaly, num_background=100):
        """
        Samples a background set for SHAP KernelExplainer using influence weights.
        Parameters:
        - x_anomaly: array-like, the instance to explain
        - num_background: int, number of samples to use as background
        Returns:
        - background_set: array-like, sampled background set
        - influence_weights: array-like, influence weights for the background set
        """
        rng = np.random.default_rng(seed=42)

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
        indices = rng.choice(len(X_train), size=num_background, p=influence_weights, replace=True)
        background_set = X_train[indices]

        return background_set
    
    def _sample_fixed_background_set(self, num_background=100):
        """
        Samples a fixed background set for SHAP KernelExplainer using uniform sampling.
        This should be called once and reused for all SHAP explanations.

        Parameters:
        - num_background: int, number of samples to use as background

        Returns:
        - background_set: array-like, sampled background set
        """
        rng = np.random.default_rng(seed=42)
        
        # Ensure X_train is 2D and convert to numpy array
        X_train = np.asarray(self.X_train)  # ensure array
        n_train = X_train.shape[0]

        # Sample background set
        if num_background >= n_train:
            background_set = X_train  # use entire training set if small
        else:
            indices = rng.choice(n_train, size=num_background, replace=False)
            background_set = X_train[indices]

        return background_set
    
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

    def _explain_with_kernelshap(self, x_anomaly, model, params, background_set=None, nsamples=500, noise_multiplier=None):
        """
        Applies SHAP KernelExplainer to anomaly scores from a detector.

        Parameters:
        - x_anomaly: array-like, the instance to explain
        - model: AnomalyDetector object, the anomaly detection model
        - params: dict, parameters for the model
        - background_set: array-like, the background set to use for SHAP
        - nsamples: int, number of samples to use for SHAP
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

        # Define local prediction wrapper using TensorFlow model
        def predict_fn(X):
            X_tensor = tf.convert_to_tensor(X.astype(np.float32))
            X_recon = model(X_tensor, training=False).numpy()
            errors = self._compute_reconstruction_error(X, X_recon, params)
            return np.sum(errors[:, top_features], axis=1)

        # Fit SHAP KernelExplainer
        explainer = shap.KernelExplainer(model=predict_fn, data=background_set)
        shap_values = explainer.shap_values(x_anomaly, silent=True, gc_collect=True, nsamples=nsamples)

        return shap_values
    
    def _explain(self, version=None, background_size=100, nsamples=500, background_method="iw"):
        """
        Main method to explain the model using SHAP KernelExplainer.

        Parameters:
        - version: str, version of the model to be explained
        - background_size: int, number of samples to use as background
        - nsamples: int, number of samples to use for SHAP
        - background_method: str, method to sample the background set (default is "iw"). Options are "fixed" or "iw".

        Returns:
        - None, but saves the SHAP values to a CSV file
        """
        # Get the test data and the corresponding data
        X_test = self.X_test.values

        # Initialize the file to save the results
        os.makedirs(f"results/explainability/{self.model_type}", exist_ok=True)
        file_path = f"results/explainability/{self.model_type}/{version}_{background_method}.csv"

        # Load the model and parameters
        model, params = self._version_info_extract(version=version)

        if self.continue_run:
            n_done = len(pd.read_csv(file_path)) if os.path.exists(file_path) else 0
        else:
            with open(file_path, "w") as f:
                f.write(",".join(self.all_cols))
                f.write("\n")
            n_done = 0

        # Sample background set
        if background_method == "fixed":
            background_set = self._sample_fixed_background_set(num_background=background_size)
        else:
            background_set = None

        # Loop through each anomaly and compute SHAP values
        for x_anomaly in tqdm(X_test[n_done : n_done + 1000], desc=f"Explaining anomalies for {self.model_type} model version: {version}"):
            # Compute the SHAP values for the current anomaly
            if background_method == "iw":
                background_set = self._sample_iw_background_set(x_anomaly, num_background=background_size)
            shap_values = self._explain_with_kernelshap(x_anomaly, model, params, background_set=background_set, nsamples=nsamples)

            # Save the SHAP values to a CSV file
            with open(file_path, "a") as f:
                f.write(",".join([str(val) for val in shap_values.reshape(-1)]))
                f.write("\n")

            # Manually free memory
            del shap_values
            gc.collect()
        
        if n_done + 1000 >= len(X_test):
            # Convert the CSV file to Feather format
            pd.read_csv(file_path).to_feather(file_path.replace(".csv", ".feather"))
            # Remove the CSV file after conversion
            os.remove(file_path)

    def explain_all_versions(self, background_size=100, nsamples=500, background_method="iw"):
        """
        Explains all versions of the model using SHAP KernelExplainer.

        Parameters:
        - background_size: int, number of samples to use as background
        - nsamples: int, number of samples to use for SHAP
        - background_method: str, method to sample the background set (default is "iw"). Options are "fixed" or "iw".

        Returns:
        - None, but saves the SHAP values for each version to a CSV file
        """
        # Loop through each version and explain
        for version in self.model_info["version"].unique():
            if os.path.exists(f"results/explainability/{self.model_type}/{version}_{background_method}.feather") and not os.path.exists(f"results/explainability/{self.model_type}/{version}_{background_method}.csv") and self.continue_run:
                print(f"SHAP values for version {version} already computed. Skipping...")
            else:
                self._explain(version=version, background_size=background_size, nsamples=nsamples, background_method=background_method)
                break

def shap_gap(baseline_shap, dp_shap, gap_type="euclidean"):
    """
    Computes the ShapGAP distance between two sets of SHAP values.

    Parameters:
    - baseline_shap: np.ndarray of shape (n_samples, n_features), SHAP values without DP
    - dp_shap: np.ndarray of shape (n_samples, n_features), SHAP values with DP
    - gap_type: str, type of distance to compute (default is "euclidean"). Options are "euclidean" or "cosine".

    Returns:
    - float, average distance across all samples
    """
    # Ensure input shapes are compatible
    assert baseline_shap.shape == dp_shap.shape, "SHAP value arrays must have the same shape."

    # Compute distance per sample
    if gap_type == "euclidean":
        shapgap_dist = np.linalg.norm(baseline_shap - dp_shap, axis=1)
    elif gap_type == "cosine":
        # Compute dot product and norms for each sample
        dot_products = np.sum(baseline_shap * dp_shap, axis=1)
        norms_before = np.linalg.norm(baseline_shap, axis=1)
        norms_after = np.linalg.norm(dp_shap, axis=1)

        # Avoid division by zero
        denom = norms_before * norms_after + 1e-10

        # Cosine similarity and distance
        cosine_sim = dot_products / denom
        shapgap_dist = 1.0 - cosine_sim
    else:
        raise ValueError("Invalid type. Use 'euclidean' or 'cosine'.")

    # Return average distance
    return shapgap_dist

def shap_length(shap: np.ndarray, threshold=0.9) -> np.ndarray:
    """
    Computes SHAP Length (SL) for each sample in a dataset.

    Parameters:
    - shap: np.ndarray of shape (n_samples, n_features), SHAP values
    - threshold: float in (0, 1), cumulative mass threshold (e.g., 0.9 for 90%)

    Returns:
    - np.ndarray of shape (n_samples,), SL_{th%} values for each sample
    """
    n_samples, n_features = shap.shape
    shap_lengths = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        phi = np.abs(shap[i])
        total_mass = np.sum(phi)
        ordering = np.argsort(-phi)
        cum_norm_mass = np.cumsum(phi[ordering]) / total_mass

        # Find how many top features are needed to reach threshold
        for j in range(n_features):
            if cum_norm_mass[j] > threshold:
                shap_lengths[i] = j + 1  # 1-based count
                break
        else:
            shap_lengths[i] = n_features

    return shap_lengths

def normalize_shap(shap_values: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    Normalize SHAP values to a consistent scale for fair comparison across models.

    Parameters:
    - shap_values: np.ndarray of shape (n_samples, n_features)
    - method: str, one of ["l2", "l1", "minmax", "zscore"]
        - "l2": normalize each SHAP vector to unit L2 norm (per sample)
        - "l1": normalize each SHAP vector to sum to 1 (L1 norm, per sample)
        - "minmax": scale each feature (column) to [0, 1] using MinMaxScaler
        - "zscore": apply z-score standardization per feature

    Returns:
    - np.ndarray of same shape, normalized SHAP values
    """
    eps = 1e-10  # small constant to avoid division by zero

    if method == "l2":
        norms = np.linalg.norm(shap_values, axis=1, keepdims=True)
        return shap_values / (norms + eps)

    elif method == "l1":
        abs_vals = np.abs(shap_values)
        sums = np.sum(abs_vals, axis=1, keepdims=True)
        return abs_vals / (sums + eps)

    elif method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(shap_values)

    elif method == "zscore":
        mean = np.mean(shap_values, axis=0, keepdims=True)
        std = np.std(shap_values, axis=0, keepdims=True)
        return (shap_values - mean) / (std + eps)

    else:
        raise ValueError(f"Unknown normalization method: '{method}'. Choose from ['l2', 'l1', 'minmax', 'zscore'].")