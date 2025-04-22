# Analyzing the Explainability-Privacy Trade-off in Differentially Private Deep Learning for Anomaly Detection

This repository contains the code, experiments, and outputs for my master's thesis.

## 📝 Description

This project investigates how differentially private mechanisms, particularly **DP-SGD** and **post-hoc output perturbation**, impact both **performance** and **explainability** in **deep learning-based anomaly detection** using autoencoders. Explainability is assessed using **KernelSHAP**.

## ⚙️ Components

- **Jupyter Notebooks (`notebooks/`)**  
  Step-by-step workflow from data exploration to evaluation:
  - `01_EDA.ipynb` – Initial exploratory data analysis  
  - `02_Preprocessing.ipynb` – Feature engineering and dataset preparation  
  - `03_Baseline_Model.ipynb` – Training and evaluating the non-private model  
  - `04_DPSGD_Model.ipynb` – Training and evaluating the DP-SGD model  
  - `05_PostHoc_DP_Model.ipynb` – Training and evaluating the output-perturbed model
  - `06_Explainability.ipynb` – Generating and analyzing SHAP explanations  
  - `07_Results_Analysis.ipynb` – Comparative analysis across models

- **Execution Scripts (`scripts/`)**  
  Run-ready Python scripts for training:
  - `baseline_model.py` – Launches training pipeline for the baseline autoencoder  
  - `dpsgd_model.py` – Launches training pipeline for the DP-SGD model

  **Basic Usage:**

  ```bash
  python scripts/baseline_model.py
  python scripts/dpsgd_model.py --epsilon 1 --delta 1e-5 --tune_method bayesian
  python scripts/dpsgd_model.py --version 202504191621 --continue_run True
  ```
  Model outputs and logs are stored in the `models/` and `logs/` directories. Hyperparameters are logged in `hyperparams/`.

- **Experiments (`experiments/`)**  
  Organized outputs from tuning and evaluation runs:

  - `hyperparam_tune/` – Stores detailed tuning results for each model:
    - `baseline/` – Tuning logs and trials for the non-private model  
    - `dpsgd/` – Tuning logs and trials for the DP-SGD model  

  - `perf_summary/` – Best performance metrics across experiments  
  - `tracking/` – Records of privacy accounting and training dynamics, including loss per epoch and cumulative privacy spending

## 📊 Results

Final outputs (on the test set) including performance metrics and SHAP-based explainability scores can be found in the `results/` directory.