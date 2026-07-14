import joblib
import pandas as pd
model = joblib.load("../models/models.pkl")

pipeline = joblib.load("../models/pipeline.pkl")

test_data = pd.read_csv("../data/test.csv")

features = test_data.drop("median_house_value", axis =1)

prepared = pipeline.transform(features)


predictions = model.predict(prepared)

test_data["Predicted_price"] = predictions

test_data.to_csv("../data/predictions.csv", index = False)


print("Predictions saved successfully.")