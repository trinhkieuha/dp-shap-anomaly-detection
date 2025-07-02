# Analyzing the Explainability-Privacy Trade-off in Differentially Private Deep Learning for Anomaly Detection

This repository contains the code, experiments, and outputs for my analysis on how DP-SGD training affects deep anomaly detection in terms or explainability.

## 📝 Description

This project investigates how **DP-SGD** impacts both **performance** and **explainability** in **deep learning-based anomaly detection** using autoencoders. Explainability is assessed using **KernelSHAP**.

## ⚙️ Components

- **Jupyter Notebooks (`notebooks/`)**  
  Step-by-step workflow from data exploration to evaluation:
  - `01_EDA.ipynb` – Initial exploratory data analysis.
  - `02_Preprocessing.ipynb` – Feature engineering and dataset preparation.
  - `03_Baseline_Model.ipynb` – Training and evaluating the non-private model.
  - `04_DPSGD_Model.ipynb` – Training and evaluating the DP-SGD model.
  - `05_Results_Analysis.ipynb` – Comparative analysis of performance metrics across models.
  - `06_Explainability.ipynb` – Analyzing SHAP explanations.

- **Execution Scripts (`scripts/`)**  
  
  **Python Scripts:**
  Run-ready Python scripts for training:
  - `baseline_model.py` – Launches hyperparameter tuning pipeline for the baseline autoencoder. Logs are saved in `logs/baseline_tune_log.txt`.
  - `dpsgd_model.py` – Launches hyperparameter tuning pipeline for the DP-SGD model. Logs are saved in `logs/dpsgd_tune_log.txt`.
  - `evaluate.py` - Trains the models using the final hyperparameter settings and evaluates performance and explainability metrics across multiple runs. If `explain=True`, explanation generation is also repeated multiple times.
  - `explain.py` - Generates explanations for each model variant. The `test=True` argument is only relevant when `evaluate.py` is run with `explain=True` to perform multiple-run evaluation. Logs are saved in `logs/explain_log.txt`.
  - `explain_eval.py` - Evaluate the explanations generated for each model variant. The `test=True` argument is only relevant when `evaluate.py` is run with `explain=True` to perform multiple-run evaluation. Logs are saved in `logs/explain_eval_log.txt`.
  - _Note_: All user-defined functions and classes used by these scripts are located in the `src/` directory.

  **Bash Scripts:**
  Executable bash scripts for experimentation and evaluation:
  - `dpsgd.sh` – Trains DP-SGD models from scratch.
  - `dpsgd_continue.sh` – Continues training of DP-SGD models from saved checkpoints.
  - `explain.sh` – Generates SHAP explanations for trained models using default settings.
  - `explain_test.sh` – Generates SHAP explanations when setting `explain=True` for `explain.py`.
  - `explain_eval.sh` – Computes explanation quality metrics.
  - `explain_eval_test.sh` – Computes explanation quality metrics when setting `explain=True` for `explain.py`.

- **Experiments (`experiments/`)**  
  Organized outputs from tuning and evaluation runs:
  - `hyperparam_tune/` – Detailed tuning results for each model.
  - `perf_summary/` – Best performance metrics across experiments.
  - `predictions/` - Test set predictions, including reconstructed inputs and binary predictions indicating whether each instance is classified as anomalous or not.
  - `scores/` - Anomaly scores of the test set.
  - `tracking/` – Records of privacy accounting and training dynamics, including loss per epoch and cumulative privacy spending.

## 📊 Results

Final outputs (on the test set) including input description, performance metrics and SHAP-based explainability scores can be found in the `results/` directory.
- **`preprocessing/`**: Summary of the input features used for modeling.
- **`metrics/`**: Performance metrics for each model.
- **`stats_eval/`**: Multiple-run evaluation results.
- **`explainability/`**: Explanation-related outputs, including SHAP values and evaluation metrics.
- **`figures/`**: Stores plots and visualizations generated during the analysis.
