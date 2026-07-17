#  California House Price Prediction using Machine Learning

Predict California house prices using Machine Learning by building an end-to-end regression pipeline with data preprocessing, feature engineering, model evaluation, and prediction.

---

##  Project Overview

This project demonstrates a complete Machine Learning workflow for predicting California housing prices using the California Housing Dataset.

The project includes:

- Data preprocessing using Scikit-Learn Pipelines
- Stratified Train-Test Split
- Feature Scaling and One-Hot Encoding
- Model Training
- Cross Validation
- Model Comparison
- Model Persistence using Joblib
- Prediction on unseen data

The primary objective is to compare multiple regression algorithms and identify the best-performing model based on Cross Validation RMSE.

---

##  Dataset

Dataset: California Housing Dataset

The dataset contains housing information such as:

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

Target Variable:

```
median_house_value
```

---

#  Project Structure

```
house-price-prediction-ml/

│

├── data/
│   ├── housing.csv
│   └── test.csv
│
├── models/
│   ├── model.pkl
│   └── pipeline.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── screenshots/
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

#  Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- Joblib

---

#  Machine Learning Workflow

```
Dataset

↓

Data Cleaning

↓

Feature Engineering

↓

Stratified Train-Test Split

↓

Preprocessing Pipeline

↓

Model Training

↓

Cross Validation

↓

Model Comparison

↓

Random Forest Selection

↓

Model Saving

↓

Prediction
```

---

#  Models Implemented

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

#  Model Performance

| Model | Training RMSE | Cross Validation RMSE |
|--------|--------------:|----------------------:|
| Linear Regression | 69050.56 | 69218.45 |
| Decision Tree | 0.00 | 70629.24 |
| Random Forest | **18342.37** | **49941.73** |

### Final Model

 Random Forest Regressor

Reason:

- Lowest Cross Validation RMSE
- Better Generalization
- Reduced Overfitting
- Best Overall Performance

---

#  Features

- End-to-End Machine Learning Pipeline
- Stratified Data Splitting
- Automated Data Preprocessing
- Feature Scaling
- One-Hot Encoding
- Cross Validation
- Multiple Model Comparison
- Model Persistence using Joblib
- Prediction Pipeline

---

#  Installation

##  Clone Repository

```bash
git clone https://github.com/Vishal7-hub/house-price-prediction-ml.git
```

---

##  Move into Project Directory

```bash
cd house-price-prediction-ml
```

---

##  Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
```

Activate Virtual Environment

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

##  Install Required Packages

```bash
pip install -r requirements.txt
```

---

##  Train the Model

```bash
cd src

python train.py
```

This will:

- Load dataset
- Split train and test data
- Build preprocessing pipeline
- Train Random Forest model
- Evaluate model
- Save trained model and preprocessing pipeline

---

##  Predict House Prices

```bash
python predict.py
```

Prediction results will be generated for unseen data.

---

#  Future Improvements

- Hyperparameter Tuning using GridSearchCV
- Feature Selection
- XGBoost
- LightGBM
- Model Deployment using FastAPI
- Docker Containerization
- CI/CD Pipeline

---

#  Key Learnings

During this project I learned:

- Building reusable preprocessing pipelines
- Handling numerical and categorical features
- Stratified Train-Test Splitting
- Cross Validation
- Model Evaluation using RMSE
- Model Comparison
- Random Forest Regression
- Model Serialization using Joblib

---

#  Author

**Vishal Kumar Roy**

Aspiring Machine Learning Engineer | Data Science Enthusiast

GitHub:
https://github.com/Vishal7-hub



---

##  If you found this project useful, consider giving it a Star.