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
from joblib import Parallel, delayed, cpu_count

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
        model = tf.keras.models.load_model(f"models/{self.model_type}/{version}_final")

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
    
    def _sample_fixed_background_set(self, num_background=200):
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
            errors[:, self.real_idx] = 0.5 * (x_r - xhat_r) ** 2 * params["gamma"]

        # Compute cross-entropy for binary features
        if self.binary_idx:
            x_b = x_anomaly[:, self.binary_idx]
            xhat_b = x_recon[:, self.binary_idx]
            xhat_b = np.clip(xhat_b, eps, 1 - eps)
            errors[:, self.binary_idx] = -x_b * np.log(xhat_b) - (1 - x_b) * np.log(1 - xhat_b) * (1 - params["gamma"])

        return errors
    
    def _predict_fn(self, model, params, top_features):
        """
        Returns a prediction function that computes anomaly scores using selected top features.

        Parameters:
        - model: the trained TensorFlow anomaly detection model
        - params: dictionary of model parameters
        - top_features: list or array of feature indices to include in the anomaly score

        Returns:
        - predict_fn: Callable that takes input X and returns anomaly scores
        """
        
        def predict_fn(X):
            X_tensor = tf.convert_to_tensor(X.astype(np.float32))
            X_recon = model(X_tensor, training=False).numpy()
            errors = self._compute_reconstruction_error(X, X_recon, params)

            return np.sum(errors[:, top_features], axis=1)
        
        return predict_fn
    
    def _compute_top_features(self, x_anomaly, model, params):
        """
        Computes the top features contributing to the anomaly score.

        Parameters:
        - x_anomaly: array-like, the instance to explain
        - model: AnomalyDetector object, the anomaly detection model
        - params: dict, parameters for the model

        Returns:
        - top_features: list of indices of the top features
        """
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

        return top_features

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
        
        # Get top features
        top_features = self._compute_top_features(x_anomaly, model, params)

        # Create a prediction function
        predict_fn = self._predict_fn(model, params, top_features)

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
        explain_size = 500
        # Get the test data and the corresponding data
        X_test = self.X_test.values

        # Initialize the file to save the results
        os.makedirs(f"results/explainability/{self.model_type}", exist_ok=True)
        file_path = f"results/explainability/{self.model_type}/{version}_{background_method}_final.csv"

        # Load the model and parameters
        model, params = self._version_info_extract(version=version)

        if os.path.exists(file_path) and self.continue_run:
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
        for x_anomaly in tqdm(X_test[n_done : n_done + explain_size], desc=f"Explaining anomalies for {self.model_type} model version: {version}"):
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
        
        if n_done + explain_size >= len(X_test):
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
            if os.path.exists(f"results/explainability/{self.model_type}/{version}_{background_method}_final.feather") and not os.path.exists(f"results/explainability/{self.model_type}/{version}_{background_method}_final.csv") and self.continue_run:
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
    - np.ndarray, distance across all samples
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

    return shapgap_dist

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
    
class ExplainabilityEvaluator:
    def __init__(self, model_type='baseline', metric='F1-Score', background_method='iw', continue_run=False):
        """
        Initializes the ExplainabilityEvaluator with model type and loads test data.
        Parameters:
        - model_type: str, type of the model to be used (default is 'baseline')
        - metric: str, tuning objective used to select the best model (default is 'F1-Score')
        - background_method: str, method to sample the background set (default is 'iw'). Options are "fixed" or "iw".
        - continue_run: bool, whether to continue a previous run (default is False)
        """
        # Set the model type and other parameters
        self.model_type = model_type
        self.metric = metric
        self.background_method = background_method
        self.continue_run = continue_run

        # Initialize SHAP KernelExplainer
        self.shap_init = ShapKernelExplainer(model_type=model_type, metric=metric, continue_run=False)

        # Find the background size and number of samples from the log file
        config = self._get_best_run_config(model_type=self.model_type, metric=self.metric, background_method=self.background_method)
        self.background_size = config.get("background_size", 100)  # Default to 100 if not found
        self.nsamples = config.get("nsamples", 100)  # Default to 100 if not found

    def _get_best_run_config(self, model_type, metric, background_method):
        """
        Extracts the best run configuration from the log file.
        Parameters:
        - model_type: str, type of the model
        - metric: str, tuning objective used to select the best model
        - background_method: str, method used for background sampling
        Returns:
        - best_config: dict, best configuration found in the log
        """

        log_path = "logs/explain_log.txt"
        with open(log_path, "r") as f:
            log = f.read()

        # Split log entries
        entries = log.split("=== Run started at:")
        best_config = None
        for entry in reversed(entries):  # Reverse to get most recent successful run first
            if "Status: SUCCESS" not in entry:
                continue

            args_match = re.search(r"Arguments:\s*(.*)", entry)
            if not args_match:
                continue

            args_line = args_match.group(1)
            args_dict = dict(
                arg.split("=") for arg in args_line.strip().split(", ") if "=" in arg
            )

            if (
                args_dict.get("model_type") == model_type
                and args_dict.get("metric") == metric
                and args_dict.get("background_method") == background_method
            ):
                best_config = {
                    "background_size": int(args_dict["background_size"]),
                    "nsamples": int(args_dict["nsamples"])
                }
                break

        return best_config

    def _noisy_baseline_perturbation(self, x, sigma=0.1, baseline="zero", n_samples=100) -> np.ndarray:
        """
        Generates a perturbation vector I = x - z0, where z0 = baseline + Gaussian noise.

        Parameters:
        - x: np.ndarray, input point x ∈ ℝ^d
        - sigma: float, standard deviation of Gaussian noise
        - baseline: str, 'zero' or 'mean' (default 'zero')

        Returns:
        - I: np.ndarray, perturbation vector
        """
        d = x.shape[1]
        # Generate all perturbations in one batch: shape (n_samples, d)
        if baseline == "zero":
            x0 = np.zeros((n_samples, d))
        elif baseline == "mean":
            x0 = np.mean(x, axis=1, keepdims=True).repeat(d, axis=1)
        else:
            raise ValueError("Unsupported baseline type.")

        noise = np.random.normal(0.0, sigma, size=(n_samples, d))
        z0 = x0 + noise
        return x - z0  # shape (n_samples, d)

    def _compute_infidelity(self, x, shap, predict_fn, original_score, n_samples=100):
        """
        Computes the infidelity of SHAP values using perturbation.
        Parameters:
        - x: array-like, the instance to explain
        - shap: np.ndarray, SHAP values for the instance
        - predict_fn: function, prediction function for the model
        - original_score: float, original score of the input sample
        - n_samples: int, number of samples to use for perturbation
        Returns:
        - float, average infidelity across all samples
        """
        I = self._noisy_baseline_perturbation(x, n_samples=n_samples)  # shape (n_samples, d)
        
        # Compute predictions for x - I
        x_perturbed = x - I  # shape (n_samples, d)
        fx_perturbed = predict_fn(x_perturbed)  # shape (n_samples,)

        # Compute errors vectorized
        dot_products = I @ shap  # shape (n_samples,)
        errors = (dot_products - (original_score - fx_perturbed)) ** 2

        return np.mean(errors)

    def _compute_sensitivity(self, x, shap, model, params, background_set, r=1e-2, n_samples=20):
        """
        Computes the sensitivity of SHAP values using perturbation.
        Parameters:
        - x: array-like, the instance to explain
        - shap: np.ndarray, SHAP values for the instance
        - model: the model used for predictions
        - params: parameters for the model
        - background_set: np.ndarray, background dataset for SHAP
        - r: float, standard deviation of Gaussian noise
        - n_samples: int, number of samples to use for perturbation

        Returns:
        - float, maximum sensitivity across all samples
        """
        def compute_single(delta):
            x_perturbed = x + delta
            if not np.isfinite(x_perturbed).all():
                return 0.0
            phi_y = self.shap_init._explain_with_kernelshap(x_perturbed, model, params,
                                                            background_set=background_set,
                                                            nsamples=self.nsamples)
            return np.linalg.norm(phi_y - shap)

        # Generate perturbations
        deltas = np.random.normal(0, r, size=(n_samples, *x.shape))

        # Run in parallel
        diffs = Parallel(n_jobs=min(8, cpu_count()))(
            delayed(compute_single)(delta) for delta in deltas
        )

        return np.max(diffs)
    
    def _compute_aopc(self, x, shap, predict_fn, original_score, n_steps=50, patch_size=1):
        """
        Computes the Area Over the Perturbation Curve (AOPC) using the MoRF strategy.

        Parameters:
        - x: np.ndarray, input sample to explain
        - shap: np.ndarray, SHAP values corresponding to the input
        - predict_fn: function, prediction function for the model
        - original_score: float, original score of the input sample
        - n_steps: int, number of perturbation steps
        - patch_size: int, number of features to perturb at each step

        Returns:
        - float, AOPC value
        """
        # Get the absolute SHAP values
        phi = np.abs(shap)
        # Sort the features by their absolute SHAP values
        order = np.argsort(-phi)  # Most relevant features first
        
        # Compute the number of perturbation steps
        perturbation_steps = min(n_steps, len(x) // patch_size) # Ensure we don't exceed the number of features
        scores = []

        # Prepare all perturbation masks
        x_replicated = np.repeat(x, perturbation_steps + 1, axis=0)

        for i in range(1, perturbation_steps + 1):
            indices_to_perturb = order[:i * patch_size]
            x_replicated[i, indices_to_perturb] = 0.0

        # Get model scores in batch
        scores_batch = predict_fn(x_replicated)

        # Compute score differences
        scores = original_score - scores_batch

        return np.mean(scores)

    def _compute_shap_length(self, shap, threshold=0.85):
        """
        Computes SHAP Length (SL) for each sample in a dataset.

        Parameters:
        - shap: np.ndarray of shape (n_features), SHAP values
        - threshold: float in (0, 1), cumulative mass threshold (e.g., 0.9 for 90%)

        Returns:
        - int, the SHAP Length (SL) for the given threshold
        """
        phi = np.abs(shap)
        total_mass = np.sum(phi)
        ordering = np.argsort(-phi)
        cum_norm_mass = np.cumsum(phi[ordering]) / (total_mass + 1e-10)

        for j, cum_mass in enumerate(cum_norm_mass):
            if cum_mass > threshold:
                return j + 1  # 1-based index

        return len(shap)
    
    def _evaluate(self, version):
        """
        Evaluates the explainability of SHAP values using infidelity and sensitivity.

        Parameters:
        - x: array-like, the instance to explain
        - n_samples: int, number of samples to use for evaluation

        Returns:
        - infidelity: float, average infidelity across all samples
        - sensitivity: float, maximum sensitivity across all samples
        """
        eval_size = 500

        # Load the model and parameters
        model, params = self.shap_init._version_info_extract(version=version)

        # Sample background set
        if self.background_method == "fixed":
            background_set = self.shap_init._sample_fixed_background_set(num_background=self.background_size)
        else:
            background_set = None
        
        # Load SHAP values
        shap_values = pd.read_feather(f"results/explainability/{self.model_type}/{version}_{self.background_method}.feather").values
        
        # Get the test data and the corresponding data
        X_test = self.shap_init.X_test.values

        # Initialize the file to save the results
        os.makedirs(f"results/explainability/{self.model_type}", exist_ok=True)
        file_path = f"results/explainability/{self.model_type}/{version}_{self.background_method}_eval.csv"

        if os.path.exists(file_path) and self.continue_run:
            n_done = len(pd.read_csv(file_path)) if os.path.exists(file_path) else 0
        else:
            with open(file_path, "w") as f:
                f.write("infidelity,aopc,shap_length")
                #f.write("sensitivity")
                f.write("\n")
            n_done = 0

        j = n_done

        # Loop through each anomaly and compute SHAP values
        for x in tqdm(X_test[n_done : n_done + eval_size], desc=f"Evaluating the explainability for {self.model_type} model version: {version}"):
            x = np.atleast_2d(x)  # ensure shape (1, d)
            
            # Compute the original score
            top_features = self.shap_init._compute_top_features(x, model, params)
            predict_fn = self.shap_init._predict_fn(model, params, top_features)
            original_score = predict_fn(x)[0]

            # Compute AOPC
            aopc = self._compute_aopc(x, shap_values[j], predict_fn=predict_fn, original_score=original_score, n_steps=50, patch_size=1)
            #print("AOPC:", aopc)
            
            # Compute infidelity
            infidelity = self._compute_infidelity(x, shap_values[j], predict_fn=predict_fn, original_score=original_score)
            #print("Infidelity:", infidelity)

            # Compute sensitivity
            #sensitivity = self._compute_sensitivity(x, shap_values[j], model, params, background_set, r=1e-2)
            #print("Sensitivity:", sensitivity)

            # Compute SHAP Length
            shap_length = self._compute_shap_length(shap_values[j])
            #print("SHAP Length:", shap_length)

            # Save the results
            with open(file_path, "a") as f:
                f.write(f"{infidelity},{aopc},{shap_length}\n")
                #f.write(f"{sensitivity}\n")

            # Manually free memory
            del infidelity, aopc, shap_length
            #del sensitivity
            gc.collect()
            j += 1

        if n_done + eval_size >= len(X_test):
            # Convert the CSV file to Feather format
            pd.read_csv(file_path).to_feather(file_path.replace(".csv", ".feather"))
            # Remove the CSV file after conversion
            os.remove(file_path)

    def evaluate_all_versions(self):
        """
        Evaluates all versions of the model using infidelity and sensitivity.

        Parameters:
        - n_samples: int, number of samples to use for evaluation

        Returns:
        - None, but saves the evaluation results for each version to a CSV file
        """
        # Loop through each version and evaluate
        for version in self.shap_init.model_info["version"].unique():
            if os.path.exists(f"results/explainability/{self.model_type}/{version}_{self.background_method}_eval.feather") and not os.path.exists(f"results/explainability/{self.model_type}/{version}_{self.background_method}_eval.csv") and self.continue_run:
                print(f"Evaluation results for version {version} already computed. Skipping...")
            else:
                self._evaluate(version=version)
                break