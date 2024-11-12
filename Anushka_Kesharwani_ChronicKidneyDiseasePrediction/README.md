# Chronic Kidney Disease Prediction Using Machine Learning

## Project Overview
The Chronic Kidney Disease Prediction project aims to predict whether a person has chronic kidney disease (CKD) based on various medical attributes. The project employs several machine learning algorithms to build models and compares their performance to select the best model for predicting CKD. The project utilizes data preprocessing, model training, hyperparameter tuning, and model evaluation to ensure accurate predictions.

## Problem Statement:
Chronic kidney disease is a major health concern affecting millions of people worldwide. Early detection can lead to better management and treatment outcomes. This project focuses on predicting whether a patient has CKD based on medical features like blood pressure, blood sugar levels, age, and more.

## Purpose
The primary purpose of this project is to:
1. Develop a model capable of predicting whether a person has chronic kidney disease based on their medical records.
2. Compare the performance of several machine learning algorithms and select the best model for predicting CKD.
3. Provide a reusable, saved model that can be used to predict new cases of CKD.

## Methods
The project follows these key steps:

### 1. Data Preprocessing:
- **Load Dataset**: Import data into a Pandas DataFrame.
- **Handle Missing Data**: Impute or remove missing values.
- **Encode Categorical Variables**: Convert categories to numerical values using one-hot encoding.
- **Split Data**: Divide the dataset into features (X) and target (y), then perform an 80-20 split for training and testing.

### 2. Model Training and Evaluation:
- **Train multiple models**: Logistic Regression, LDA, KNN, Decision Tree, and Random Forest.
- **Evaluation Metrics**: Use accuracy, precision, recall, and F1 score to compare model performance.
- **Model Selection**: Choose the Random Forest model due to its high recall.

### 3. Hyperparameter Tuning:
- Apply GridSearchCV to optimize the Random Forest model's hyperparameters for better performance.

### 4. Model Saving:
- Save the best-performing Random Forest model using joblib for future predictions.

## Results
The results of the project are as follows:
- **Performance Metrics**: Each model's performance is evaluated based on accuracy, precision, recall, and F1 score. The Random Forest Classifier showed the highest recall, which is crucial for detecting CKD cases.
- **Best Model**: Random Forest Classifier, after hyperparameter tuning, is the chosen model for predicting CKD due to its higher performance in recall.
- **Final Model**: The best Random Forest model is saved and can be loaded later for making predictions on new data.

## Requirements
-jupyter
- pandas
- scikit-learn
- joblib



