# Model Pipeline

This repository contains an Airflow DAG (Directed Acyclic Graph) for a model pipeline. The pipeline includes tasks for data extraction, transformation, training, prediction, and evaluation of Industry orders.

You can find the original development script in [modelo.ipynb](https://github.com/viniciusfjacinto/machine-learning-fiec/blob/main/modelo.ipynb) and the Airflow DAG in [ml_dag.py](https://github.com/viniciusfjacinto/machine-learning-fiec/blob/main/dags/ml_dag.py)

The file [dados.csv](https://github.com/viniciusfjacinto/machine-learning-fiec/blob/main/dados.csv) contains the data used in this project.

## Table of Contents

- [Introduction](#introduction)
- [Tasks](#tasks)
  - [Extract Data](#extract-data)
  - [Transform Data](#transform-data)
  - [Train Model](#train-model)
  - [Predict](#predict)
  - [Evaluate Model](#evaluate-model)
- [Usage](#usage)

## Introduction

The purpose of this pipeline is to automate the process of analyzing, exploring, visualizing data, and creating a predictive model for the "Profit" column of the provided dataset. The pipeline leverages Airflow for orchestrating the workflow, allowing seamless execution and monitoring of each task.

### Pipeline Structure
![image](https://github.com/viniciusfjacinto/machine-learning-fiec/assets/87664450/a940ed7e-9586-48cd-95cc-2fb94a65fe7f)

### Prediction Results
![image](https://github.com/viniciusfjacinto/machine-learning-fiec/assets/87664450/b6b0badc-1166-496e-95bb-42f47f2d122f)

## Tasks

### Extract Data
- **Process**:
  - Connects to an AWS Athena database using provided credentials.
  - Executes an SQL query to retrieve the dataset (`raw.fiec_industry_data`).
  - Returns the extracted data for further processing.

### Transform Data
- **Process**:
  - Preprocesses the extracted data:
    - Converts column names to title case.
    - Creates a binary categorical column "Profit_Class" based on whether Profit is greater than zero.
    - Encodes categorical variables such as "Customer_Name" and "Product_Name".
    - Performs feature engineering, including creating "Encoded_Customer" and "Encoded_Product" columns based on customer and product counts.
    - One-hot encodes remaining categorical columns.
    - Sets "Order_Id" as the index and drops irrelevant columns.
  - Returns the preprocessed data.

### Train Model
- **Process**:
  - Receives the preprocessed data from the previous task.
  - Divides the data into features (X) and target (y).
  - Identifies and removes highly correlated columns to avoid multicollinearity.
  - Splits the data into training and testing sets.
  - Selects the top 20 features using SelectKBest with f_regression scoring function.
  - Trains a Support Vector Regression (SVR) model with a linear kernel using the selected features.
  - Returns the filtered training features, testing features, and target variables.

### Predict
- **Process**:
  - Receives the filtered training and testing features along with target variables.
  - Utilizes the trained SVR model to predict the target variable on the testing dataset.
  - Calculates the Mean Squared Error (MSE) between the predicted and actual values.
  - Returns the predicted values and actual values in DataFrame format.

### Evaluate Model
- **Process**:
  - Receives the predicted and actual values from the previous task.
  - Calculates the percentage difference between the predicted and actual values.
  - Counts the number of predictions with less than 20% difference from the actual values.
  - Calculates the percentage of satisfactory predictions.
  - Returns the percentage of satisfactory predictions.

