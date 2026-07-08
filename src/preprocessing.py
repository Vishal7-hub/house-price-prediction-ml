from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing_pipeline(numerical_columns , categorical_columns):
    num_pipeline = Pipeline([
        ("imputer" , SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
            ])

    
    cat_pipeline= Pipeline([
        ("onehot" , OneHotEncoder(handle_unknown ="ignore")),

    ])

    full_pipeline = ColumnTransformer([
        ("num" , num_pipeline , numerical_columns),
        ("cat",cat_pipeline,categorical_columns)

    ])

    return full_pipeline

