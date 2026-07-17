#  House Price Prediction using Machine Learning

An end-to-end Machine Learning project for predicting California housing prices using Scikit-Learn.

#  California House Price Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)

![License](https://img.shields.io/badge/License-MIT-green)

## Tech Stack

- Python
- NumPy
- Pandas
- Scikit-Learn
- Joblib
- Matplotlib

## Project Structure

```text
house-price-prediction-ml/
│
├── data/
├── models/
├── notebooks/
├── screenshots/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

This project uses the **California Housing Dataset**, containing housing-related features such as location, median income, population, and house value.

Target Variable:

- `median_house_value`




##  Model Training and Evaluation

Three machine learning regression models were trained and evaluated to predict California housing prices.

### Models Implemented

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Evaluation Strategy

To obtain a reliable estimate of model performance, 5-Fold Cross Validation was used instead of relying only on training performance.

**Evaluation Metric**

- Root Mean Squared Error (RMSE)

### Model Performance

| Model | Training RMSE | Cross Validation RMSE | Standard Deviation |
|--------|--------------:|----------------------:|-------------------:|
| Linear Regression | 69050.56 | 69218.45 | 689.50 |
| Decision Tree | 0.00 | 70629.24 | 1447.66 |
| Random Forest | **18342.37** | **49941.73** | **760.90** |

### Key Findings

- Linear Regression provided a stable baseline but underfit the dataset.
- Decision Tree achieved perfect training performance but suffered from severe overfitting.
- Random Forest delivered the lowest cross-validation RMSE, making it the best-performing model.
- Cross-validation confirmed that Random Forest generalized significantly better than the other models.

### Selected Model

 Random Forest Regressor was selected as the final model due to its superior predictive performance and better generalization on unseen data.


 ##  Model Persistence

The final Random Forest model and preprocessing pipeline are serialized using Joblib for future inference.

### Saved Artifacts

- `models/model.pkl` – Trained Random Forest model
- `models/pipeline.pkl` – Preprocessing pipeline

### Prediction Workflow

1. Load trained model
2. Load preprocessing pipeline
3. Read unseen input data
4. Apply preprocessing
5. Generate predictions
6. Save results to `predictions.csv`

This approach eliminates the need to retrain the model for every prediction and follows standard machine learning deployment practices.

> Project is currently under development.