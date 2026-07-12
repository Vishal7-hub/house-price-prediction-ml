import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from preprocessing import build_preprocessing_pipeline
from sklearn.model_selection import cross_val_score


housing = pd.read_csv("../data/housing.csv")

print(housing.head())
print()

print(housing.info())
print()

print(housing.describe())


housing["income_cat"]= pd.cut(
    housing["median_income"],
    bins = [0.0 ,1.5, 3.0, 4.5, 6.0 ,np.inf],
    labels = [1,2,3,4,5]
)


split = StratifiedShuffleSplit(
    n_splits= 1,
    test_size =0.2,
    random_state=42

)

for train_index , test_index in split.split(housing , housing["income_cat"]):
    train_set = housing.loc[train_index].drop("income_cat" , axis = 1)
    test_set = housing.loc[test_index].drop("income_cat" , axis =1)


test_set.to_csv("../data/test.csv", index=False)

print(f"Training samples : {len(train_set)}")
print(f"Testing samples : {len(test_set)}")

train_labels = train_set["median_house_value"].copy()
train_features = train_set.drop("median_house_value" , axis =1)


numerical_columns = train_features.drop("ocean_proximity" , axis =1).columns.tolist()
categorical_columns = ["ocean_proximity"]


pipeline = build_preprocessing_pipeline(numerical_columns , categorical_columns)

train_prepared = pipeline.fit_transform(train_features)

print(train_prepared.shape)



# Linear Regression

linear_model = LinearRegression()

linear_model.fit(train_prepared, train_labels)

linear_predictions = linear_model.predict(train_prepared)

linear_rmse = root_mean_squared_error(
    train_labels,
    linear_predictions
)

print(f"Linear Regression RMSE : {linear_rmse:.2f}")

linear_scores = cross_val_score(
    linear_model,
    train_prepared,
    train_labels,
    scoring="neg_root_mean_squared_error",
    cv=5
)
linear_rmse_scores = -linear_scores
print("Linear Regression CV RMSE:", linear_rmse_scores.mean())
print("Standard Deviation:", linear_rmse_scores.std())




# Decision Tree

tree_model = DecisionTreeRegressor(
    random_state=42
)

tree_model.fit(
    train_prepared,
    train_labels
)

tree_predictions = tree_model.predict(
    train_prepared
)

tree_rmse = root_mean_squared_error(
    train_labels,
    tree_predictions
)

print(f"Decision Tree RMSE : {tree_rmse:.2f}")

tree_scores = cross_val_score(
    tree_model,
    train_prepared,
    train_labels,
    scoring="neg_root_mean_squared_error",
    cv=5
)

tree_rmse_scores = -tree_scores

print("Decision Tree CV RMSE:", tree_rmse_scores.mean())

print("Standard Deviation:", tree_rmse_scores.std())