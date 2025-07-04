import re
from datetime import datetime
import pandas as pd
import tensorflow as tf
import pickle
from src.models import AutoencoderTrainer, AnomalyDetector
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import pi
import os
from src.dp_utils import *
from src.eda import data_info
from tqdm import tqdm
import random
import subprocess
from src.explainability import shap_gap, normalize_shap

plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # or other, 'dejavuserif'
plt.rcParams['font.family'] = 'serif'  # or 'DejaVu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 'DejaVu Serif' serif' 'Times'

class ValidationEvaluation:
    def __init__(self, X_val, y_val, real_cols, binary_cols, all_cols, dp_sgd=False, post_hoc=False):
        """
        Initialize the ValidationEvaluation class.
        Parameters:
        - X_val: pd.DataFrame, validation feature set
        - y_val: pd.Series, validation labels
        - real_cols: list of str, names of real-valued columns
        - binary_cols: list of str, names of binary columns
        - all_cols: list of str, names of all columns
        - dp_sgd: bool, whether to use differential privacy settings
        - post_hoc: bool, whether to use post-hoc evaluation
        """

        self.X_val = X_val
        self.y_val = y_val
        self.real_cols = real_cols
        self.binary_cols = binary_cols
        self.all_cols = all_cols
        self.dp_sgd = dp_sgd
        self.post_hoc = post_hoc

        self.metric_labels = {
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-Score",
            "auc": "AUC"
        }
        self.metrics = list(self.metric_labels.keys())

    def extract_latest_successful_bayesian_versions(self, log_path):
        """
        Extract the latest successful Bayesian versions from the log file.
        Parameters:
        - log_path: str, path to the log file
        Returns:
        - latest_versions: dict, mapping of metric to version and tuning strategy
            (for dp_sgd, also includes epsilon and delta)
        """
        with open(log_path, "r") as f:
            log_content = f.read()

        # Split log into blocks
        log_blocks = log_content.strip().split("=== Run started at:")

        # Track latest runs
        latest_versions = {}

        for block in log_blocks[1:]:
            lines = block.strip().splitlines()
            args_line = next((l for l in lines if l.startswith("Arguments:")), None)
            end_line = next((l for l in lines if l.startswith("Run ended at:")), None)
            status_line = next((l for l in lines if l.startswith("Status:")), None)

            if not args_line or not end_line or not status_line:
                continue  # Skip incomplete runs

            # Parse values
            version_match = re.search(r"version=(\d+)", args_line)
            metric_match = re.search(r"metric=(\w+)", args_line)
            epsilon_match = re.search(r"epsilon=([\d\.]+)", args_line) if self.dp_sgd or self.post_hoc else None
            delta_match = re.search(r"delta=([\deE\.\-]+)", args_line) if self.dp_sgd or self.post_hoc else None
            noise_mechanism_match = re.search(r"noise_mechanism=(\w+)", args_line) if self.post_hoc else None
            end_time = re.search(r"Run ended at: (.*)", end_line).group(1).strip()
            status = re.search(r"Status: (\w+)", status_line).group(1).strip()

            if not (version_match and metric_match and status == "SUCCESS"):
                continue

            version = version_match.group(1)
            metric = metric_match.group(1)
            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f")

            if self.dp_sgd or self.post_hoc:
                epsilon = float(epsilon_match.group(1)) if epsilon_match else None
                if epsilon < 1:
                    continue
                delta = float(delta_match.group(1)) if delta_match else None
                if self.post_hoc:
                    noise_mechanism = noise_mechanism_match.group(1) if noise_mechanism_match else None
                    key = (metric, epsilon, delta, noise_mechanism)
                else:
                    key = (metric, epsilon, delta)
                if key not in latest_versions or end_dt > latest_versions[key]["end_time"]:
                    if self.post_hoc:
                        latest_versions[key] = {
                            "version": version,
                            "end_time": end_dt,
                            "metric": metric,
                            "epsilon": epsilon,
                            "delta": delta,
                            "noise_mechanism": noise_mechanism
                        }
                    else:
                        latest_versions[key] = {
                            "version": version,
                            "end_time": end_dt,
                            "metric": metric,
                            "epsilon": epsilon,
                            "delta": delta
                        }
            else:
                if metric not in latest_versions or end_dt > latest_versions[metric][1]:
                    latest_versions[metric] = (version, end_dt)

        if self.dp_sgd:
            return {
                v["version"]: (self.metric_labels.get(v["metric"], v["metric"]), v["epsilon"], v["delta"], v["end_time"])
                for v in latest_versions.values()
            }
        elif self.post_hoc:
            return {
                v["version"]: (self.metric_labels.get(v["metric"], v["metric"]), v["epsilon"], v["delta"], v["noise_mechanism"], v["end_time"])
                for v in latest_versions.values()
            }
        else:
            return {
                version: (self.metric_labels[metric], end_time)
                for metric, (version, end_time) in latest_versions.items()
            }

    def evaluate_model_performance(self, version_info_dict):
        """
        Evaluate anomaly detection models on a validation set.

        Parameters:
        - version_info_dict: dict, mapping of model versions to tuning strategies and metrics
            (for dp_sgd, also includes epsilon and delta)
        Returns:
        - eval_results: pd.DataFrame, evaluation results with columns for metrics and tuning strategy
        """
        eval_results = pd.DataFrame(columns=["version"] + self.metrics)

        # Determine model and hyperparameter paths
        if self.dp_sgd:
            model_path_prefix="../models/dpsgd"
            hyperparam_path_prefix="../hyperparams/dpsgd"
        elif self.post_hoc:
            model_path_prefix="../models/posthoc_dp"
            hyperparam_path_prefix="../hyperparams/posthoc_dp"
        else:
            model_path_prefix="../models/baseline"
            hyperparam_path_prefix="../hyperparams/baseline"

        for version in version_info_dict.keys():
            print(f"Evaluating version {version}")

            model_info = version_info_dict[version]
            if self.dp_sgd:
                metric, epsilon, delta, end_dt = model_info
                noise_mechanism = None
                print(f"Metric: {metric}, Epsilon: {epsilon}, Delta: {delta}")
            elif self.post_hoc:
                metric, epsilon, delta, noise_mechanism, end_dt = model_info
                print(f"Metric: {metric}, Epsilon: {epsilon}, Delta: {delta}, Noise Mechanism: {noise_mechanism}")
            else:
                metric, end_dt = model_info
                epsilon, delta = 0, 0
                noise_mechanism = None
                print(f"Metric: {metric}")

            # Load model and hyperparameters
            model = tf.keras.models.load_model(f"{model_path_prefix}/{version}")
            with open(f"{hyperparam_path_prefix}/{version}.pkl", "rb") as f:
                params = pickle.load(f)

            detector = AnomalyDetector(
                model=model,
                real_cols=self.real_cols,
                binary_cols=self.binary_cols,
                all_cols=self.all_cols,
                lam=params["lam"],
                gamma=params["gamma"],
                post_hoc=self.post_hoc,
                noise_mechanism=noise_mechanism,
                target_epsilon=float(epsilon), delta=float(delta)
            )

            # Compute scores on validation set
            if self.post_hoc:
                scores = pd.read_feather(f"../experiments/scores/posthoc_dp/{version}.feather")
            else:
                scores = detector._compute_anomaly_scores(self.X_val)

            y_pred = detector._detect(scores, params["threshold"])
            perf = detector._evaluate(y_pred, self.y_val, scores)
            perf.update(params)
            perf["version"] = str(version)
            perf["tuned_by"] = metric
            if self.dp_sgd or self.post_hoc:
                perf["delta"] = delta
                perf["epsilon"] = epsilon
                if self.post_hoc:
                    perf["noise_mechanism"] = noise_mechanism
            perf["end_time"] = end_dt
            eval_results = pd.concat([eval_results, pd.DataFrame([perf])], ignore_index=True)

        eval_results.set_index("version", inplace=True)

        # Ensure 'tuned_by' is treated as a categorical column with the desired order
        eval_results['tuned_by'] = pd.Categorical(eval_results['tuned_by'], categories=self.metric_labels.values(), ordered=True)

        # Then sort the DataFrame accordingly
        eval_results = eval_results.sort_values(by='tuned_by')
        
        return eval_results

    def plot_bar(self, eval_results, save=True):
        """
        Plots barplots of validation performance grouped by tuning strategy.

        Parameters:
        - eval_results: pd.DataFrame, evaluation results with columns for metrics and tuning strategy
        - save: bool, whether to save the plots
        """
        # Determine grouping
        groups = [((None, None), eval_results)] if not self.dp_sgd and not self.post_hoc else eval_results.groupby(["epsilon", "delta"] if self.dp_sgd else ["epsilon", "delta", "noise_mechanism"])

        for model_info, group_df in groups:
            if self.dp_sgd:
                eps, delt = model_info
            elif self.post_hoc:
                eps, delt, noise_mechanism = model_info
            else:
                eps, delt = None, None

            # Create subplots
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
            axs = axs.flatten()

            for i, metric in enumerate(self.metrics):
                ax = axs[i]
                bars = ax.bar(group_df['tuned_by'], group_df[metric], color="skyblue")
                ax.set_title(self.metric_labels.get(metric, metric))
                ax.set_xlabel("Tuned by")
                ax.set_ylabel("")

                y_min = group_df[metric].min()
                y_max = group_df[metric].max()
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.05
                #ax.set_ylim(y_min - margin, y_max + margin)
                ax.set_ylim(0, y_max + 0.1)
                ax.tick_params(axis='x', rotation=45)
                ax.bar_label(bars, fmt="%.4f", padding=3)

            # Remove unused axes
            for j in range(len(self.metrics), len(axs)):
                fig.delaxes(axs[j])

            # Title and layout
            title = "Validation Performance"
            if self.dp_sgd or self.post_hoc:
                title += r" ($\varepsilon$ = " + str(eps) + r", $\delta$ = " + str(delt) + ")"
                if self.post_hoc:
                    title += f" - {noise_mechanism.capitalize()}"
            fig.suptitle(title, fontsize=16)
            plt.tight_layout()

            # Save or show
            if save is True:
                filename = f"../results/figures/dpsgd_hyperparam_tune_eps{eps}_delta{delt}.png" if self.dp_sgd else f"../results/figures/posthoc_dp_hyperparam_tune_eps{eps}_delta{delt}_{noise_mechanism}.png" if self.post_hoc else f"../results/figures/baseline_hyperparam_tune.png"
                plt.savefig(filename)
            
            plt.show()

    def plot_radar(self, eval_results):
        """
        Plots radar charts of validation performance grouped by tuning strategy.

        Parameters:
        - eval_results: pd.DataFrame, evaluation results with columns for metrics and tuning strategy
        - save: bool, whether to save the plots
        """
        # Determine grouping
        groups = [((None, None), eval_results)] if not self.dp_sgd else eval_results.groupby(["epsilon", "delta"])

        # Determine the angles
        num_metrics = len(self.metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        # Number of columns for subplots
        cols = 2

        for (eps, delt), group_df in groups:
            # Number of models and metrics
            num_models = len(group_df)

            # Determine the number of rows needed
            rows = int(np.ceil(num_models / cols))

            # Create subplots
            fig, axs = plt.subplots(rows, cols, subplot_kw=dict(polar=True), figsize=(5 * cols, 4 * rows))
            axs = axs.flatten()  # Flatten in case of single row
            j = 0

            # Plot each model
            for i, row in group_df.iterrows():
                values = row[self.metrics].tolist()
                values += values[:1]
                ax = axs[j]
                ax.plot(angles, values, marker='o')
                ax.fill(angles, values, alpha=0.25)
                ax.set_title(f"Tuned by {row['tuned_by']}", size=10)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(self.metric_labels.values())
                for label,i in zip(ax.get_xticklabels(),range(0,len(angles))):

                    angle_rad=angles[i]
                    if angle_rad <= pi/2:
                        ha= 'left'
                        va= "bottom"

                    elif pi/2 < angle_rad <= pi:
                        ha= 'right'
                        va= "bottom"

                    elif pi < angle_rad <= (3*pi/2):
                        ha= 'right'
                        va= "top"  

                    else:
                        ha= 'left'
                        va= "bottom"

                    label.set_verticalalignment(va)
                    label.set_horizontalalignment(ha) 
                ax.set_ylim(0, 1)
                ax.grid(True)
                j += 1

            # Hide any unused subplots
            for k in range(j, len(axs)):
                fig.delaxes(axs[k])
            
            # Title and layout
            title = "Validation Performance"
            if self.dp_sgd:
                title += r" ($\varepsilon$ = " + str(eps) + r", $\delta$ = " + str(delt) + ")"
            fig.suptitle(title, fontsize=16)
            plt.tight_layout()

            plt.show()


    def plot_heatmap(self, eval_results):
        """
        Plots heatmaps of validation performance grouped by tuning strategy.
        Parameters:
        - eval_results: pd.DataFrame, evaluation results with columns for metrics and tuning strategy
        """
        # Determine grouping
        groups = [((None, None), eval_results)] if not self.dp_sgd else eval_results.groupby(["epsilon", "delta"])
        
        for (eps, delt), group_df in groups:
            # Set up the dataframe for heatmap
            heatmap_data = group_df[self.metrics]

            # Plot heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f", cbar=True)
            plt.ylabel("Tuned by")
            plt.xlabel("Performance")
            plt.xticks(ticks=list(np.arange(0.5, len(self.metrics), 1)), labels=self.metric_labels.values())
            plt.yticks(ticks=list(np.arange(0.5, len(heatmap_data), 1)), labels=group_df.tuned_by)
            
            # Title and layout
            title = "Validation Performance"
            if self.dp_sgd:
                title += r" ($\varepsilon$ = " + str(eps) + r", $\delta$ = " + str(delt) + ")"
            plt.title(title, fontsize=16)
            plt.tight_layout()

            plt.show()

def compute_fidelity(baseline_prediction, prediction):
    """
    Computes the fidelity of a model's predictions.
    Parameters:
    - baseline_prediction: np.ndarray, baseline model predictions
    - prediction: np.ndarray, new model predictions
    Returns:
    - fidelity: float, fidelity score
    """
    return np.mean(baseline_prediction == prediction)

class StatisticalEval():
    def __init__(self, explain=False):

        self.explain = explain

        # Load training data
        self.X_train = pd.read_feather("data/processed/X_train.feather")

        # Load train-validation data
        self.X_train_val = pd.read_feather("data/processed/X_train_validate.feather")
        
        # Load test data
        self.X_test = pd.read_feather("data/processed/X_test.feather")
        self.y_test = pd.read_feather("data/processed/y_test.feather")
        
        # Extract variable types from metadata
        self.var_info = data_info(self.X_test)
        self.all_cols = self.X_test.columns
        self.real_cols = self.var_info[self.var_info["var_type"] == "numerical"]["var_name"].tolist()
        self.binary_cols = self.var_info[self.var_info["var_type"] == "binary"]["var_name"].tolist()

        # Metrics
        self.metric_labels = {
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-Score",
            "auc": "AUC"
        }

    def _single_eval(self, model_type, version, epsilon, delta, seed=None):
        
        with open(f"hyperparams/{model_type}/{version}.pkl", "rb") as f:
            params = pickle.load(f)
        
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Train model
        trainer = AutoencoderTrainer(
            input_dim=self.X_train.shape[1],
            real_cols=self.real_cols,
            binary_cols=self.binary_cols,
            all_cols=self.all_cols,
            activation='relu',
            patience_limit=10,
            verbose=False,
            dp_sgd=True if model_type == "dpsgd" else False,
            post_hoc=False,
            target_epsilon=epsilon,
            delta=delta,
            version=version,
            save_tracking=True,
            raise_convergence_error=True,
            **{key: value for key, value in params.items() if key not in ['threshold', 'q']}
        )
        model = trainer.train(self.X_train, self.X_train_val)

        detector = AnomalyDetector(
                model=model,
                real_cols=self.real_cols,
                binary_cols=self.binary_cols,
                all_cols=self.all_cols,
                lam=params['lam'],
                gamma=params['gamma'],
            )
        scores, x_hat = detector._compute_anomaly_scores(self.X_test, test_set=True)

        # Save reconstructed data
        pd.DataFrame(x_hat, columns=self.all_cols).to_feather(f"experiments/predictions/{model_type}/{version}_recons_final.feather")

        # Detect
        y_pred = detector._detect(scores, params['threshold'])

        # Save predictions
        pd.DataFrame(y_pred, columns=["anomaly"]).to_feather(f"experiments/predictions/{model_type}/{version}_pred_final.feather")

        return model

    def _multi_eval(self, model_list, seeds=None, n_runs=100):
        
        # If seeds are not specified, set from 0 to n_runs
        if not seeds:
            seeds = np.arange(n_runs)

        # Initialize params_dict and min_runs
        params_dict = {}
        min_runs = n_runs # The min number of runs across models

        # Loop over all models to get params_dict and min_runs
        for model in model_list:
            model_type = model[0]
            version = model[1]
            # Load model and hyperparameters
            with open(f"hyperparams/{model_type}/{version}.pkl", "rb") as f:
                params_dict[model] = pickle.load(f)

            # Initialize results storage
            if self.explain:
                log_path = f"results/stats_eval/{model_type}/{version}_test.csv"
            else:
                log_path = f"results/stats_eval/{model_type}/{version}.csv"
            os.makedirs(os.path.dirname(log_path), exist_ok=True) # Create directory if it doesn't exist
            if not os.path.exists(log_path):
                with open(log_path, "w") as f:
                    if model_type == 'baseline':
                        f.write(",".join(self.metric_labels.values()) + "\n")
                    else:
                        f.write(",".join(self.metric_labels.values()) + ",Fidelity\n")
                if self.explain:
                    agg_type = ["mean", "std", "median", "iqr"]
                    if model_type == 'baseline':
                        explain_metrics = ["infidelity", "aopc"]
                    else:
                        explain_metrics = ["infidelity", "aopc", "shapgap_euclidean", "shapgap_cosine"]
                    metric_col = [f"{a}_{metric}"for metric in explain_metrics for a in agg_type]
                    with open(f"results/stats_eval/{model_type}/{version}_eval.csv", "w") as f:
                        f.write(",".join(metric_col) + "\n")
                exist_runs = 0
            else:
                if self.explain:
                    exist_runs = min([len(pd.read_csv(log_path)), len(pd.read_csv(f"results/stats_eval/{model_type}/{version}_eval.csv"))])
                else:
                    exist_runs = len(pd.read_csv(log_path))
            
            # If the exist runs of the current model < current min_runs
            if exist_runs < min_runs:
                min_runs = exist_runs

        # Delete the results of incomplete runs
        for model in model_list:
            model_type = model[0]
            version = model[1]
            if self.explain:
                log_path = f"results/stats_eval/{model_type}/{version}_test.csv"
            else:
                log_path = f"results/stats_eval/{model_type}/{version}.csv"
            pd.read_csv(log_path)[:min_runs].to_csv(log_path, index=False)
            pd.read_csv(f"results/stats_eval/{model_type}/{version}_eval.csv")[:min_runs].to_csv(f"results/stats_eval/{model_type}/{version}_eval.csv", index=False)
        
        # Run multiple evaluations
        for r in tqdm(range(min_runs, n_runs), desc=f"Evaluating models"):
            tf.random.set_seed(seeds[r])
            np.random.seed(seeds[r])
            random.seed(seeds[r])
            
            # Train and evaluate prediction
            for (model_type, version, epsilon, delta), params in params_dict.items():
                if self.explain:
                    log_path = f"results/stats_eval/{model_type}/{version}_test.csv"
                else:
                    log_path = f"results/stats_eval/{model_type}/{version}.csv"
                # Train model
                trainer = AutoencoderTrainer(
                    input_dim=self.X_train.shape[1],
                    real_cols=self.real_cols,
                    binary_cols=self.binary_cols,
                    all_cols=self.all_cols,
                    activation='relu',
                    patience_limit=10,
                    verbose=False,
                    dp_sgd=True if model_type == "dpsgd" else False,
                    post_hoc=False,
                    target_epsilon=epsilon,
                    delta=delta,
                    save_tracking=False,
                    raise_convergence_error=True,
                    **{key: value for key, value in params.items() if key not in ['threshold', 'q']}
                )
                model = trainer.train(self.X_train, self.X_train_val)
                if self.explain: # Save only for explanability
                    model.save(f"models/{model_type}/{version}_test")

                # Initialize anomaly detector
                detector = AnomalyDetector(
                    model=model,
                    real_cols=self.real_cols,
                    binary_cols=self.binary_cols,
                    all_cols=self.all_cols,
                    lam=params['lam'],
                    gamma=params['gamma'],
                )

                # Compute scores
                scores = detector._compute_anomaly_scores(self.X_test)

                # Detect
                threshold = np.quantile(scores, params['q'])
                y_pred = detector._detect(scores, threshold)

                if model_type == 'baseline':
                    baseline_y_pred = y_pred
                else:
                    # Compute fidelity
                    fidelity = compute_fidelity(baseline_y_pred, y_pred)

                # Evaluate
                metrics = detector._evaluate(y_pred, self.y_test, scores)
                metric_values = [metrics[metric] for metric in self.metric_labels.keys()]

                # Log results
                with open(log_path, "a") as f:
                    perf = ",".join([f"{value}" for value in metric_values])
                    if model_type == "baseline":
                        perf += "\n"
                    else:
                        perf += f",{fidelity}" + "\n"
                    f.write(perf)

            # Explain
            if self.explain:
                subprocess.run(["bash", "scripts/explain_test.sh"])
                subprocess.run(["bash", "scripts/explain_eval_test.sh"])

                # Evaluate the explanations
                for (model_type, version, epsilon, delta), params in params_dict.items():
                    # Write to the log file
                    eval_log_path = f"results/stats_eval/{model_type}/{version}_eval.csv"
                    
                    # Read the explanationa and corresponding evaluation
                    explanation = pd.read_feather(f"results/explainability/{model_type}/{version}_fixed_test.feather")
                    explain_eval = pd.read_feather(f"results/explainability/{model_type}/{version}_fixed_eval_test.feather")

                    # Statistical properties of the evaluation
                    agg_type = ["mean", "std", "median", "iqr"]
                    if model_type == 'baseline':
                        explain_metrics = ["infidelity", "aopc"]
                    else:
                        explain_metrics = ["infidelity", "aopc", "shapgap_euclidean", "shapgap_cosine"]
                    metric_col = [f"{a}_{metric}"for metric in explain_metrics for a in agg_type]
                    eval_dict = {}
                    for metric in explain_metrics[:2]:
                        eval_dict[f"mean_{metric}"] = explain_eval[metric].mean()
                        eval_dict[f"std_{metric}"] = explain_eval[metric].std()
                        eval_dict[f"median_{metric}"] = explain_eval[metric].median()
                        eval_dict[f"iqr_{metric}"] = explain_eval[metric].quantile(0.75) - explain_eval[metric].quantile(0.25)

                    # Calculate the SHAPGap
                    if model_type == "baseline":
                        baseline_explanation = explanation
                    
                    elif model_type == "dpsgd":
                        dpsgd_explanation = explanation
                        for gap_type in ["euclidean", "cosine"]:
                            # Normalize the explanations
                            baseline_explanation_norm = normalize_shap(baseline_explanation, method="l2")
                            dpsgd_explanation_norm = normalize_shap(dpsgd_explanation, method="l2")
                            # Compute SHAPGap
                            gap = shap_gap(baseline_explanation_norm, dpsgd_explanation_norm, gap_type=gap_type)
                            # Calculate the statistical properties
                            eval_dict[f"mean_shapgap_{gap_type}"] = np.mean(gap)
                            eval_dict[f"std_shapgap_{gap_type}"] = np.std(gap)
                            eval_dict[f"median_shapgap_{gap_type}"] = np.median(gap)
                            eval_dict[f"iqr_shapgap_{gap_type}"] = np.quantile(gap, 0.75) - np.quantile(gap, 0.25)

                    # Write to the log file
                    eval_list = f'{",".join([f"{eval_dict[col]:.3f}" for col in metric_col])}\n'
                    with open(eval_log_path, "a") as f:
                        f.write(eval_list)

    def __call__(self, metric_used="AUC", n_runs=100):
        
        # Load the best models
        # Baseline
        baseline = pd.read_csv("experiments/perf_summary/baseline_val_results.csv")
        baseline_model = baseline.query(f'tuned_by == "{metric_used}"')
        # DP-SGD
        dpsgd = pd.read_csv("experiments/perf_summary/dpsgd_val_results.csv")
        dpsgd_models = dpsgd.query(f'tuned_by == "{metric_used}"')

        # Check if any runs are already performed
        n_rows = 0
        for version in baseline_model["version"].tolist():
            try:
                if self.explain:
                    log_path = f"results/stats_eval/baseline/{version}_test.csv"
                else:
                    log_path = f"results/stats_eval/baseline/{version}.csv"
                n_rows += len(pd.read_csv(log_path))
            except:
                n_rows += 0
        for version in dpsgd_models["version"].tolist():
            try:
                if self.explain:
                    log_path = f"results/stats_eval/dpsgd/{version}_test.csv"
                else:
                    log_path = f"results/stats_eval/dpsgd/{version}.csv"
                n_rows += len(pd.read_csv(log_path))
            except:
                n_rows += 0
        """if n_rows == 0:
            seeds = random.sample(range(1000000), n_runs)
            with open("results/stats_eval/seeds.txt", "w") as f:
                for seed in seeds:
                    f.write(f"{seed}\n")
        else:"""
        with open("results/stats_eval/seeds.txt", "r") as f:
            seeds = [int(line.strip()) for line in f]

        model_list = []
        
        # Load baseline model versions
        for version in baseline_model["version"].tolist():
            model_list.append(("baseline", version, 0, 0))

        # Load dpsgd model versions
        for i, row in dpsgd_models.iterrows():
            epsilon = row["epsilon"]
            delta = row["delta"]
            version = row["version"]
            model_list.append(("dpsgd", version, epsilon, delta))
        
        # Run the multiple-run test
        self._multi_eval(model_list, seeds, n_runs=n_runs)

        # Find the median run
        # Calculate the median index
        median_ind = int(len(seeds)/2)

        # Get all the multiple-run test results
        perf_stats = pd.DataFrame()
        for (model_type, version, epsilon, delta) in model_list:
            # Read the result file
            perf = pd.read_csv(f"results/stats_eval/{model_type}/{version}.csv")
            # Add the model name column
            perf.insert(0, "Model", f"{model_type}/{version}")
            # Add the seed column
            perf["seed"] = seeds

            # Concat the tmp result to the final result
            perf_stats = pd.concat([perf_stats, perf], ignore_index=True)

        # Find the median seed
        mean_by_seed = perf_stats.drop(columns=["Model"]).groupby("seed").mean().sort_values(by="AUC")
        median_seed = int(mean_by_seed[median_ind:median_ind+1].index[0])

        # Run the final model
        print("Start training the final model...")
        for (model_type, version, epsilon, delta) in model_list:
            model = self._single_eval(model_type, version, epsilon, delta, median_seed)
            model.save(f"models/{model_type}/{version}_final") # Save the final model

        return print(f"Statistical evaluation completed for {n_runs} runs with metric '{metric_used}'.")