# Fraud Detection in Financial Transactions Using Autoencoders

Autoencoders are unsupervised neural networks used for anomaly detection. The idea behind using them for fraud detection is that they are trained to learn a compressed representation of the data (normal transactions). When a fraudulent transaction occurs, the reconstruction error of the autoencoder will be significantly higher, signaling that the transaction is anomalous.

## 1. Data Collection

I used [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download) dataset from kaggle.

