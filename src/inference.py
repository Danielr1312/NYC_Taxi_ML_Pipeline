import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import logging
import pandas as pd
import joblib
import lightgbm as lgb
from src.config import *
from src.data_ingestion import fetch_weather_data, extract_month_from_filename
from src.data_cleaning import *
from src.feature_engineering import add_trip_datetime_features, add_rain_or_snow_column
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_model(model_path):
    """ Load trained model """
    if not os.path.exists(model_path):
        logging.error(f"❌ Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logging.info(f"💾 Loading trained model from {model_path}...")
    model = joblib.load(model_path)
    logging.info("✅ Model loaded successfully!")
    return model

def load_preprocessor(preprocessor_path):
    """ Load the pre-trained preprocessor """
    if not os.path.exists(preprocessor_path):
        logging.error(f"❌ Preprocessor file not found at {preprocessor_path}")
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
    
    logging.info(f"💾 Loading preprocessor from {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)
    logging.info("✅ Preprocessor loaded successfully!")
    return preprocessor

def preprocess_input_data(input_data, preprocessor, api_key, weather_dir, drop_columns):
    """
    Cleans, processes, and applies the saved preprocessing pipeline to new input data.

    Parameters:
    - input_data (pd.DataFrame): Raw input data.
    - preprocessor (sklearn.pipeline.Pipeline): Pre-trained preprocessing pipeline.
    - dataset_name (str): Name of the dataset file (used to extract date).
    - api_key (str): API key for fetching weather data.
    - weather_dir (str): Directory for storing weather data.
    - drop_columns (list): Columns to drop from the dataset.

    Returns:
    - np.ndarray: Preprocessed feature matrix.
    """

    # Check if the labels column exists
    y_true = None
    label_column = "fare_amount"  # Change this to your actual label column name
    labels_available = label_column in input_data.columns

    # 🔄 Extract date from pickup_datetime
    try:
        if "pickup_datetime" in input_data.columns:
            pickup_value = input_data["pickup_datetime"].iloc[0]
        elif "tpep_pickup_datetime" in input_data.columns:
            pickup_value = input_data["tpep_pickup_datetime"].iloc[0]
        else:
            raise KeyError("pickup_datetime column not found.")

        logging.info(f"⏱️ Raw pickup datetime value: {pickup_value} (type: {type(pickup_value)})")

        # Handle numeric timestamps (assume seconds if < 10^11, else milliseconds)
        pickup_datetime = None
        if np.issubdtype(type(pickup_value), np.number):
            logging.info("🧠 Treating value as Unix timestamp in seconds.")
            pickup_datetime = pd.to_datetime(pickup_value, unit="s")
        elif isinstance(pickup_value, str):
            logging.info("🧠 Treating value as ISO 8601 string.")
            pickup_datetime = pd.to_datetime(pickup_value, errors="coerce")
        elif isinstance(pickup_value, pd.Timestamp):
            logging.info("🧠 Value is already a pandas Timestamp.")
            pickup_datetime = pickup_value
        else:
            raise ValueError("pickup_datetime value is not numeric, string, or Timestamp.")


        if pd.isna(pickup_datetime):
            raise ValueError(f"Invalid datetime format: {pickup_value}")

        dataset_month = pickup_datetime.strftime("%Y-%m")
        inferred_filename = f"yellow_tripdata_{dataset_month}.parquet"
        logging.info(f"🔄 Inferred filename month from pickup datetime: {inferred_filename}")

        # Fetch weather
        try:
            weather_data = fetch_weather_data(inferred_filename, api_key, weather_dir)
            logging.info(f"✅ Weather data loaded: {weather_data.shape}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to fetch weather data: {e}")
            weather_data = None

    except Exception as e:
        logging.warning(f"⚠️ Could not extract pickup_datetime or infer dataset month: {e}")
        weather_data = None


    # logging.info("🔄 Extracting date from dataset name...")
    # date = extract_month_from_filename(dataset_name)

    # # Attempt to fetch weather data
    # try:
    #     weather_data = fetch_weather_data(dataset_name, api_key, weather_dir)
    #     logging.info(f"✅ Weather data loaded: {weather_data.shape}")
    # except Exception as e:
    #     logging.warning(f"⚠️ Failed to fetch weather data: {e}")
    #     weather_data = None

    # ✅ Normalize datetime columns
    datetime_columns = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]

    for col in datetime_columns:
        if col in input_data.columns:
            original_dtype = input_data[col].dtype
            if np.issubdtype(original_dtype, np.number):
                # Interpret as Unix timestamp in seconds
                input_data[col] = pd.to_datetime(input_data[col], unit="s", errors="coerce")
            else:
                input_data[col] = pd.to_datetime(input_data[col], errors="coerce")

            logging.info(f"📅 Converted '{col}' to datetime (original dtype: {original_dtype})")

    # 1️⃣ Clean Data
    logging.info("🧹 Cleaning input data...")
    input_data, _ = filter_trips_by_dataset_date(input_data, inferred_filename)
    input_data, _ = remove_negative_total_amount(input_data)

    # 2️⃣ Remap Values (Ensure categorical consistency)
    logging.info("🔄 Remapping categorical values...")
    input_data['store_and_fwd_flag'] = input_data['store_and_fwd_flag'].map({'N': 0, 'Y': 1}).astype('Int32')
    input_data['RatecodeID'] = input_data['RatecodeID'].astype('Int32')
    input_data['payment_type'] = input_data['payment_type'].astype('Int32')

    input_data = convert_value_to_missing(input_data, 'passenger_count', [0])
    input_data = convert_value_to_missing(input_data, 'payment_type', [0])
    input_data = convert_value_to_missing(input_data, 'RatecodeID', [99])
    input_data = convert_value_to_missing(input_data, 'VendorID', [6])
    input_data = remap_unknown_payment_types(input_data)

    # 3️⃣ Feature Engineering (Ensure feature consistency)
    logging.info("🚀 Engineering features...")
    input_data = add_trip_datetime_features(input_data, drop_original=False)

    input_data, _ = remove_long_trips(input_data, max_hours=MAX_TRIP_DURATION_HOURS)
    input_data, _ = remove_short_trips(input_data, min_minutes=MIN_TRIP_DURATION_MINUTES)
    input_data = add_estimated_speed(input_data)
    input_data, _ = remove_high_speed_trips(input_data, max_mph=MAX_ESTIMATED_SPEED_MPH)
    input_data, _ = remove_low_speed_trips(input_data, min_mph=MIN_ESTIMATED_SPEED_MPH)

    if weather_data is not None:
        input_data = add_rain_or_snow_column(input_data, weather_data)

    # Separate features and labels (if available)
    if labels_available:
        y_true = input_data[label_column]
    
    # Drop unnecessary columns
    input_data.drop(columns=drop_columns, inplace=True)

    if input_data.empty:
        logging.warning("🚫 No valid rows left after cleaning.")
        return None, None

    logging.info("✅ Feature engineering completed.")

    # Ensure we access the ColumnTransformer inside the Pipeline
    column_transformer = preprocessor.named_steps["preprocessor"]

    # Extract categorical feature names
    categorical_cols = column_transformer.transformers_[1][2]

    # Convert categorical columns to string (if needed)
    input_data[categorical_cols] = input_data[categorical_cols].astype(str)

    logging.info("🔄 Applying preprocessing pipeline...")
    processed_array = preprocessor.transform(input_data)

    # Extract feature names from pipeline
    feature_names = preprocessor.named_steps["preprocessor"].get_feature_names_out()

    # Wrap in a DataFrame
    processed_data = pd.DataFrame(processed_array.toarray(), columns=feature_names)

    logging.info(f"✅ Preprocessing completed. Shape: {processed_data.shape}")


    return processed_data, y_true

def make_predictions(model, input_data):
    """ Generate predictions using the trained model. """
    logging.info("🎯 Making predictions...")
    predictions = model.predict(input_data)
    logging.info("✅ Predictions generated successfully!")
    return predictions

def evaluate_predictions(y_true, y_pred):
    """
    Evaluates model predictions if ground truth labels are available.

    Parameters:
    - y_true (pd.Series): True labels.
    - y_pred (pd.Series): Predicted values.

    Returns:
    - dict: Performance metrics (MAE, RMSE, R²).
    """
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,  # sqrt of MSE
        "R² Score": r2_score(y_true, y_pred)
    }
    return metrics

def main(input_file, output_file):
    """ Main function to handle inference. """
    logging.info("🚀 Starting inference pipeline...")

    # Load trained model and preprocessor
    model = load_model(MODEL_PATH)
    preprocessor = load_preprocessor(PREPROCESSOR_PATH)

    # Load input data
    logging.info(f"📂 Loading input data from {input_file}...")
    if input_file.endswith(".csv"):
        input_data = pd.read_csv(input_file)
    elif input_file.endswith(".parquet"):
        input_data = pd.read_parquet(input_file)
    else:
        logging.error("❌ Unsupported file format! Use .csv or .parquet.")
        return

    logging.info(f"✅ Input data loaded! Shape: {input_data.shape}")

    

    # Preprocess input data
    processed_data, y_true = preprocess_input_data(input_data, preprocessor, API_KEY, WEATHER_DIR, DROP_COLUMNS)

    # Make predictions
    predictions = make_predictions(model, processed_data)

    # Save predictions
    logging.info(f"💾 Saving predictions to {output_file}...")
    pd.DataFrame(predictions, columns=["predictions"]).to_csv(output_file, index=False)
    logging.info("✅ Predictions saved successfully!")

    # Evaluate model if labels are available
    if y_true is not None:
        logging.info("📊 Evaluating model performance...")
        metrics = evaluate_predictions(y_true, predictions)

        # Log the results
        logging.info(f"📈 Model Performance:\n{metrics}")

        # Optionally save metrics to a file
        metrics_file = output_file.replace(".csv", "_metrics.txt")
        with open(metrics_file, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

        logging.info(f"✅ Evaluation results saved to {metrics_file}")
    else:
        logging.info("✅ No labels in input data, inference completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new data using trained LightGBM model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input data file (CSV or Parquet).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save predictions (CSV).")

    args = parser.parse_args()
    main(args.input_file, args.output_file)

# usage example:
# python src/inference.py --input_file "data/raw/yellow_tripdata_2024-02.parquet" --output_file "data/predictions/predictions_2024-02.csv"