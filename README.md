# KNN Diabetes Classification

A machine learning project using the K-Nearest Neighbors (KNN) algorithm to classify diabetes outcomes based on patient health metrics.

## Overview

This project implements a KNN classifier to predict whether a patient has diabetes based on various health indicators. The model uses the Elbow Method to determine the optimal number of neighbors (k) for classification.

## Dataset

The dataset contains medical measurements for diabetes patients with the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age
- Outcome (target variable: 0 = no diabetes, 1 = diabetes)

## Project Structure

### 1. **Data Loading and Exploration**
   - Load diabetes dataset from CSV
   - Display first few rows
   - Show dataset information and statistics
   - Visualize feature correlations using a heatmap

### 2. **Data Preprocessing**
   - Separate features (X) and target variable (y)
   - Split data into training (70%) and testing (30%) sets
   - Standardize features using StandardScaler for consistent scaling

### 3. **Model Training**
   - Initialize KNN classifier with k=7
   - Train the model on scaled training data

### 4. **Model Evaluation**
   - Make predictions on test data
   - Generate classification report with precision, recall, and F1-score
   - Display confusion matrix

### 5. **Hyperparameter Optimization**
   - Use the Elbow Method to find optimal k value
   - Test k values from 1 to 29
   - Plot error rates to visualize performance

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Ensure you have the diabetes.csv file in the correct path: `/Users/kajal.parmar/Documents/diabetes.csv`
2. Open the Jupyter notebook: `KNN_Diabetes_Classification.ipynb`
3. Run all cells to execute the complete pipeline

## Key Results

- **Model**: K-Nearest Neighbors Classifier
- **Default K Value**: 7
- **Train-Test Split**: 70-30
- **Feature Scaling**: StandardScaler (Z-score normalization)

## Model Performance

The classification report provides:
- **Precision**: How many predicted positives are actually positive
- **Recall**: How many actual positives were correctly identified
- **F1-Score**: Harmonic mean of precision and recall

The confusion matrix shows:
- True Negatives (TN): Correctly predicted no diabetes
- False Positives (FP): Incorrectly predicted diabetes
- False Negatives (FN): Missed diabetes cases
- True Positives (TP): Correctly predicted diabetes

## Elbow Method

The Elbow Method tests k values from 1 to 29 and plots the error rate for each k. This helps identify the optimal number of neighbors that minimizes classification error without overfitting.

## Author

Kajal Parmar

## License

This project is open source and available for educational purposes.
