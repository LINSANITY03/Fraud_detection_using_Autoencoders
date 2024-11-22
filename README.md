# Fraud Detection in Financial Transactions Using Autoencoders

Autoencoders are unsupervised neural networks used for anomaly detection. The idea behind using them for fraud detection is that they are trained to learn a compressed representation of the data (normal transactions). When a fraudulent transaction occurs, the reconstruction error of the autoencoder will be significantly higher, signaling that the transaction is anomalous.

## 1. Data Collection

I used [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download) dataset from kaggle.

## Directory Structure

project/
│
├── data/
│   ├── raw/               # Raw data files (unmodified, original data)
│   ├── processed/         # Preprocessed data files ready for modeling
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Notebook for EDA (Exploratory Data Analysis)
│   ├── 02_preprocessing.ipynb     # Notebook for data cleaning and preprocessing
│   ├── 03_modeling.ipynb          # Notebook for model training
│   └── 04_evaluation.ipynb        # Notebook for model evaluation and results
│
├── scripts/
│   ├── data_processing.py   # Python script for data preprocessing
│   ├── train_model.py       # Python script for model training
│   └── evaluate_model.py    # Python script for evaluation
│
├── models/
│   ├── saved_models/        # Serialized models (e.g., .pkl, .h5)
│   └── model_logs/          # Logs and checkpoints from training
│
├── logs/
│   ├── training_logs/       # Logs for training progress
│   └── experiment_logs/     # Logs for experiments
│
├── results/
│   ├── figures/             # Visualizations, plots
│   ├── reports/             # Analysis reports, markdowns
│   └── metrics/             # Saved evaluation metrics (e.g., CSV, JSON)
│
├── requirements.txt         # Python dependencies
├── README.md                # Project overview and instructions
└── config.yaml              # Configuration file for project-wide parameters
