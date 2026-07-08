import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from preprocessing import build_preprocessing_pipeline


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
