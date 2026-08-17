# Human Activity Classification

## Overview

This project implements multiple machine learning classification models
using the UCI Human Activity Recognition Using Smartphones dataset.

The models are demonstrated through an interactive Streamlit web application.

## Dataset

- Dataset: UCI Human Activity Recognition Using Smartphones
- Classification: Multi-class
- Test instances: 2,947
- Features: 561
- Classes: 6

## Machine Learning Models

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Random Forest

## Evaluation Metrics

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Streamlit Features

The application provides:

- Test CSV upload
- Model selection
- Predictions
- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC
- Confusion Matrix
- Classification Report
- Comparison of all five models

## Project Structure

Human_Activity_Classification/

    app.py
    requirements.txt
    README.md
    test_data.csv
    model_results.csv

    model/
        scaler.pkl
        logistic_regression.pkl
        random_forest.pkl
        naive_bayes.pkl
        decision_tree.pkl
        knn.pkl

## How to Run

Install the required packages:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## GitHub Repository

To be added after GitHub upload.

## Streamlit Community Cloud

To be added after deployment.

## Model Comparison

The following table reports the actual evaluation results obtained from the
five trained classification models on the UCI Human Activity Recognition
test dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.954869 | 0.997485 | 0.956650 | 0.954869 | 0.954809 | 0.946140 |
| Decision Tree | 0.862233 | 0.917281 | 0.863273 | 0.862233 | 0.861652 | 0.834813 |
| KNN | 0.880217 | 0.976437 | 0.888331 | 0.880217 | 0.879025 | 0.857806 |
| Gaussian Naive Bayes | 0.770275 | 0.958428 | 0.794683 | 0.770275 | 0.768770 | 0.728609 |
| Random Forest | 0.926026 | 0.995126 | 0.927377 | 0.926026 | 0.925957 | 0.911330 |


## Model Observations

### 1. Logistic Regression

Logistic Regression achieved the strongest overall performance in this
experiment. It obtained an accuracy of 0.954869,
AUC of 0.997485, F1 Score of 0.954809,
and MCC of 0.946140. It is therefore the best-performing
model among the five evaluated classifiers based on the reported metrics.

### 2. Decision Tree

The Decision Tree achieved an accuracy of
0.862233
and an F1 Score of
0.861652.
Its performance is lower than Logistic Regression and Random Forest in this
experiment.

### 3. K-Nearest Neighbor

KNN achieved an accuracy of
0.880217
and an F1 Score of
0.879025.
It performed better than the Decision Tree and Gaussian Naive Bayes models,
but below Logistic Regression and Random Forest.

### 4. Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of
0.770275
and an F1 Score of
0.768770.
It produced the lowest overall classification performance among the five
models based on accuracy, F1 Score, and MCC.

### 5. Random Forest

Random Forest achieved an accuracy of
0.926026
and an F1 Score of
0.925957.
It was the second strongest model overall and demonstrated consistently
high performance across the evaluation metrics.

### Overall Comparison

Based on the experimental results, **Logistic Regression** is the best
performing model. It achieved the highest Accuracy, AUC, Precision, Recall,
F1 Score, and MCC among the five models evaluated.



Model-specific observations will be added using the actual experimental
results from the trained models.

## Deployment

The application is intended for deployment using Streamlit Community Cloud.
