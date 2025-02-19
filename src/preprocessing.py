import pandas as pd
import numpy as np
from src.config import RANDOM_STATE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

def build_preprocessing_pipeline(numerical_cols, categorical_cols):
    """
    Creates a preprocessing pipeline for taxi trip data.

    Returns:
    - sklearn.pipeline.Pipeline: A scikit-learn pipeline for preprocessing taxi data.
    """
    # Create the preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', IterativeImputer(n_nearest_features=10, max_iter=5, random_state=RANDOM_STATE)), # O(n[max_iter])
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], n_jobs=-1) # Use all available cores

    return Pipeline(steps=[('preprocessor', preprocessor)])