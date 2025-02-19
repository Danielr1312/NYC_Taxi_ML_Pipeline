import os
import json
import joblib
import tempfile
import logging
import numpy as np
from lightgbm import early_stopping
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy.stats import randint, uniform
from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, JOBLIB_BACKEND_DIR

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set joblib backend directory
if os.path.exists(JOBLIB_BACKEND_DIR):
    joblib_temp_dir = os.path.join(JOBLIB_BACKEND_DIR, 'joblib_temp')
    logging.info(f"📁 Using Joblib Temp Directory: {joblib_temp_dir}")
else:
    joblib_temp_dir = os.path.join(tempfile.gettempdir(), 'joblib_temp')
    logging.warning(f"⚠️ Joblib Backend Directory not found. Using default temp directory: {joblib_temp_dir}")

# Create joblib temp directory if it doesn't exist
os.makedirs(joblib_temp_dir, exist_ok=True)

# Set joblib temp directory
os.environ['JOBLIB_TEMP_FOLDER'] = joblib_temp_dir
logging.info(f"✅ Joblib Temp Directory Set: {os.environ['JOBLIB_TEMP_FOLDER']}")

def train_model(X_train, y_train):
    """
    Trains a LightGBM Regressor.

    Parameters:
    - X_train (array-like): Preprocessed training data.
    - y_train (array-like): Training target labels.

    Returns:
    - model: Trained model.
    """
    logging.info("🚀 Training LightGBM Model...")

    # Split for early stopping
    X_train_sub, X_valid, y_train_sub, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LGBMRegressor(
        boosting_type='gbdt',
        n_estimators=500,
        learning_rate=0.1,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X_train_sub, y_train_sub,
        eval_set=[(X_valid, y_valid)],
        eval_metric='mae',
        callbacks=[early_stopping(50)]
    )

    return model  # final_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on test data.

    Parameters:
    - model: Trained model.
    - X_test (array-like): Preprocessed test data.
    - y_test (array-like): Test target labels.

    Returns:
    - metrics (dict): Dictionary containing evaluation metrics.
    """

    y_pred = model.predict(X_test)

    # Compute Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2 Score": round(r2, 3)
    }

    logging.info(f"📊 Model Evaluation Results: {metrics}")

    return metrics
