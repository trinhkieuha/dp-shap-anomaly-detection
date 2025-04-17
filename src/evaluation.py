import re
from datetime import datetime
import pandas as pd
import tensorflow as tf
import pickle
from src.models import AnomalyDetector
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import pi

plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # or other, 'dejavuserif'
plt.rcParams['font.family'] = 'serif'  # or 'DejaVu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # 'DejaVu Serif' serif' 'Times'

class ValidationEvaluation:
    def __init__(self, X_val, y_val, real_cols, binary_cols, all_cols, dp_sgd=False):
        """
        Initialize the ValidationEvaluation class.
        Parameters:
        - X_val: pd.DataFrame, validation feature set
        - y_val: pd.Series, validation labels
        - real_cols: list of str, names of real-valued columns
        - binary_cols: list of str, names of binary columns
        - all_cols: list of str, names of all columns
        - dp_sgd: bool, whether to use differential privacy settings
        """

        self.X_val = X_val
        self.y_val = y_val
        self.real_cols = real_cols
        self.binary_cols = binary_cols
        self.all_cols = all_cols
        self.dp_sgd = dp_sgd

        self.metric_labels = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1 Score",
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
            method_match = re.search(r"tune_method=(\w+)", args_line)
            metric_match = re.search(r"metric=(\w+)", args_line)
            epsilon_match = re.search(r"epsilon=([\d\.]+)", args_line) if self.dp_sgd else None
            delta_match = re.search(r"delta=([\deE\.\-]+)", args_line) if self.dp_sgd else None
            end_time = re.search(r"Run ended at: (.*)", end_line).group(1).strip()
            status = re.search(r"Status: (\w+)", status_line).group(1).strip()

            if not (version_match and method_match and metric_match and status == "SUCCESS"):
                continue

            version = version_match.group(1)
            method = method_match.group(1)
            metric = metric_match.group(1)
            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f")

            if method != "bayesian":
                continue

            if self.dp_sgd:
                epsilon = epsilon_match.group(1) if epsilon_match else None
                delta = delta_match.group(1) if delta_match else None
                key = (metric, epsilon, delta)
                if key not in latest_versions or end_dt > latest_versions[key]["end_time"]:
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
                v["version"]: (self.metric_labels.get(v["metric"], v["metric"]), v["epsilon"], v["delta"])
                for v in latest_versions.values()
            }
        else:
            return {
                version: self.metric_labels[metric]
                for metric, (version, _) in latest_versions.items()
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

        if self.dp_sgd:
            model_path_prefix="../models/dpsgd"
            hyperparam_path_prefix="../hyperparams/dpsgd"
        else:
            model_path_prefix="../models/baseline"
            hyperparam_path_prefix="../hyperparams/baseline"

        for version in version_info_dict.keys():
            print(f"Evaluating version {version}")
            model = tf.keras.models.load_model(f"{model_path_prefix}/{version}")
            with open(f"{hyperparam_path_prefix}/{version}.pkl", "rb") as f:
                params = pickle.load(f)

            detector = AnomalyDetector(
                model=model,
                real_cols=self.real_cols,
                binary_cols=self.binary_cols,
                all_cols=self.all_cols,
                lam=params["lam"],
                gamma=params["gamma"]
            )

            scores = detector._compute_anomaly_scores(self.X_val)
            y_pred = detector._detect(scores, params["threshold"])
            perf = detector._evaluate(y_pred, self.y_val, scores)
            perf["version"] = str(version)
            eval_results = pd.concat([eval_results, pd.DataFrame([perf])], ignore_index=True)

        # Metadata: metric label, epsilon, delta
        eval_results["tuned_by"] = eval_results["version"].apply(lambda x: version_info_dict[x][0] if self.dp_sgd else version_info_dict[x])
        if self.dp_sgd:
            eval_results["epsilon"] = eval_results["version"].apply(lambda x: version_info_dict[x][1])
            eval_results["delta"] = eval_results["version"].apply(lambda x: version_info_dict[x][2])

        eval_results.set_index("version", inplace=True)
        return eval_results

    def plot_bar(self, eval_results, save=True):
        """
        Plots barplots of validation performance grouped by tuning strategy.

        Parameters:
        - eval_results: pd.DataFrame, evaluation results with columns for metrics and tuning strategy
        - save: bool, whether to save the plots
        """
        # Determine grouping
        groups = [((None, None), eval_results)] if not self.dp_sgd else eval_results.groupby(["epsilon", "delta"])

        for (eps, delt), group_df in groups:
            # Create subplots
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
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
                ax.set_ylim(y_min - margin, y_max + margin)
                ax.tick_params(axis='x', rotation=45)
                ax.bar_label(bars, fmt="%.3f", padding=3)

            # Remove unused axes
            for j in range(len(self.metrics), len(axs)):
                fig.delaxes(axs[j])

            # Title and layout
            title = "Validation Performance"
            if self.dp_sgd:
                title += f" (ε = {eps}, δ = {delt})"
            fig.suptitle(title, fontsize=16)
            plt.tight_layout()

            # Save or show
            if save is not None:
                filename = f"../results/figures/dpsgd_hyperparam_tune_eps{eps}_delta{delt}.png" if self.dp_sgd else f"../results/figures/baseline_hyperparam_tune.png"
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