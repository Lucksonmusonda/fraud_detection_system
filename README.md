# fraud_detection_system

## Project Overview
Title: Financial Fraud Detection System
Description: The Financial Fraud Detection System is a machine learning-based project designed to detect fraudulent transactions. It uses a dataset of credit card transactions to train a model that can accurately classify transactions as either fraudulent or legitimate. The system is built using Python and popular machine learning libraries like scikit-learn and XGBoost, and it includes a Flask API for easy deployment and integration into other systems.

## Description

The Financial Fraud Detection System is a machine learning project designed to detect fraudulent transactions from a dataset of credit card transactions. The model is trained to classify transactions as either fraudulent or legitimate using features such as transaction amount, time, and various anonymized numerical features. This project employs Python's popular libraries such as pandas, scikit-learn, XGBoost, and Flask for model training, evaluation, and deployment.


## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [API Deployment](#api-deployment)
- [Testing the API](#testing-the-api)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

 ## Technologies Used
Python: Programming language
pandas: Data manipulation and analysis
scikit-learn: Machine learning library
XGBoost: Extreme Gradient Boosting algorithm
imbalanced-learn: Handling imbalanced datasets
Flask: Web framework for deploying the model



## Installation

To get started with this project, clone the repository and install the required dependencies.

```bash

  ## Project Structure
  
  fraud_detection_system/
├── model/                       # Contains the trained model
├── app/                         # Flask application
│   └── app.py                   # Flask API code
├── data/                        # Contains the dataset
│   └── creditcard.csv           # Credit card transaction dataset
├── scripts/                     # Data preprocessing and training scripts
│   ├── train_model.py           # Model training script
│   └── preprocess_data.py       # Data preprocessing script
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

git clone https://github.com/luckson_musonda/fraud_detection_system.git
cd fraud_detection_system

Create a virtual environment:
python -m venv venv
Activate the virtual environment:

Windows: .\venv\Scripts\activate

Install the required libraries:
pip install -r requirements.txt

model training 
Train the machine learning model using the training script provided:

# scripts/train_model.py

import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from preprocess_data import preprocess_data

# Load and preprocess data
X_res, y_res = preprocess_data('data/creditcard.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Initialize and train the model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print('AUC-ROC:', roc_auc_score(y_test, y_proba))

Run the training script:

# Save the trained model
joblib.dump(model, 'model/fraud_detection_model.pkl')

# app/app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('../model/fraud_detection_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

Run the Flask app:
import requests

# Define the API endpoint
url = 'http://127.0.0.1:5000/predict'  # Update with your actual API endpoint

# Define multiple sets of input data to test
test_inputs = [
    {"features": [0.1, 1.5, 3.2]},  # Update with the actual features expected by your model
    {"features": [0.5, 2.0, 1.7]},
    {"features": [0.9, 0.3, 4.1]},
    # Add more sets of inputs as needed
]

# Loop through each set of inputs and test the API
for i, input_data in enumerate(test_inputs):
    response = requests.post(url, json=input_data)
    print(f"Test Input {i+1}: {input_data}")
    
    # Print status code and response content for debugging
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)
    
    try:
        # Attempt to parse the response as JSON
        print("API Response:", response.json())
    except requests.exceptions.JSONDecodeError:
        print("Failed to parse response as JSON")

    print("-" * 50)



License
This project is licensed under the MIT License - see the LICENSE file for details.

### **Summary**

This README file provides a comprehensive overview of your financial fraud detection project, including:

- **Project Description**: Brief description of the project and its purpose.
- **Installation**: Instructions for setting up the environment and installing dependencies.
- **Data Preparation and Model Training**: Steps to preprocess data and train the machine learning model.
- **API Deployment**: Guide on how to deploy the model as a Flask API.
- **Testing the API**: Instructions to test the API with example data.
- **Project Structure**: Overview of the project folder structure.
- **Technologies Used**: List of technologies and libraries used in the project.
- **Contributing**: Invitation for others to contribute to the project.
- **License**: Information about the project’s license.




