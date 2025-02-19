import os
import pandas as pd
import logging
import joblib
import json
from src.config import *
from src.data_ingestion import load_taxi_data, fetch_weather_data
from src.data_cleaning import *
from src.feature_engineering import add_trip_datetime_features, add_rain_or_snow_column
from src.preprocessing import build_preprocessing_pipeline
from src.model_training import train_model, evaluate_model
from utils.api_utils import extract_month_from_filename
from sklearn.model_selection import train_test_split

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """
    Main script entry point.
    """
    logging.info("🚀 Starting NYC Taxi Processing and Training Pipeline...\n")

    # 1️⃣ Load Raw Data
    logging.info("\n[1] Loading raw data...")
    taxi_data = load_taxi_data(TAXI_DATA_PATH)

    # Extract month from filename
    date = extract_month_from_filename(DATA_NAME)

    # Ensure directories exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Attempt to fetch weather data
    try:
        weather_data = fetch_weather_data(DATA_NAME, API_KEY, WEATHER_DIR)
        logging.info(f"✅ Weather data loaded: {weather_data.shape}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to fetch weather data: {e}")
        weather_data = None


    # 2️⃣ Clean Data
    logging.info("\n[2] Cleaning data...")

    # 2.1 Filtering & Removing Data
    taxi_data, _ = filter_trips_by_dataset_date(taxi_data, DATA_NAME)
    taxi_data, _ = remove_negative_total_amount(taxi_data)

    logging.info(f"✅ Data cleaned: {taxi_data.shape}")

    # 3️⃣ Remapping Values
    logging.info("\n[3] Remapping categorical values...")
    taxi_data['store_and_fwd_flag'] = taxi_data['store_and_fwd_flag'].map({'N': 0, 'Y': 1}).astype('Int32')
    taxi_data['RatecodeID'] = taxi_data['RatecodeID'].astype('Int32')
    taxi_data['payment_type'] = taxi_data['payment_type'].astype('Int32')

    taxi_data = convert_value_to_missing(taxi_data, 'passenger_count', [0])
    taxi_data = convert_value_to_missing(taxi_data, 'payment_type', [0])
    taxi_data = convert_value_to_missing(taxi_data, 'RatecodeID', [99])
    taxi_data = convert_value_to_missing(taxi_data, 'VendorID', [6])
    taxi_data = remap_unknown_payment_types(taxi_data)

    

    # Save Cleaned Data
    cleaned_data_path = os.path.join(PROCESSED_DATA_DIR, f"taxi_data_cleaned_{date}.parquet")
    taxi_data.to_parquet(cleaned_data_path)
    logging.info(f"✅ Saved cleaned data to {cleaned_data_path}")

    # 4️ Engineer Features
    logging.info("\n[4] Engineering features...")
    taxi_data = add_trip_datetime_features(taxi_data, drop_original=False)

    # 4.1 Filtering Data by Date
    taxi_data, _ = remove_long_trips(taxi_data, MAX_TRIP_DURATION_HOURS)
    taxi_data, _ = remove_short_trips(taxi_data, MIN_TRIP_DURATION_MINUTES)
    taxi_data = add_estimated_speed(taxi_data)
    taxi_data, _ = remove_high_speed_trips(taxi_data, MAX_ESTIMATED_SPEED_MPH)
    taxi_data, _ = remove_low_speed_trips(taxi_data, MIN_ESTIMATED_SPEED_MPH)

    if weather_data is not None:
        taxi_data = add_rain_or_snow_column(taxi_data, weather_data)

    # 3.1 Dropping Columns
    taxi_data.drop(columns=DROP_COLUMNS, inplace=True)
        

    # Save Engineered Data
    engineered_data_path = os.path.join(PROCESSED_DATA_DIR, f"taxi_data_engineered_{date}.parquet")
    taxi_data.to_parquet(engineered_data_path)
    logging.info(f"✅ Saved feature-engineered data to {engineered_data_path}")

    

    # 5️⃣ Preprocessing
    logging.info("\n[5] Preprocessing data...")

    # 5.1 Convert Categorical Features to Strings for Preprocessing
    for column in CATEGORICAL_FEATURES:
        taxi_data[column] = taxi_data[column].astype('str')

    # 5.2 Splitting Data into Features and Target
    X = taxi_data.drop(columns=[TARGET_FEATURE])
    y = taxi_data[TARGET_FEATURE]

    # 5.3 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    logging.info(f"✅ Training data: {X_train.shape}, Testing data: {X_test.shape}")

    # Save Train-Test Data
    X_train.to_parquet(os.path.join(PROCESSED_DATA_DIR, f"train_data_{date}.parquet"))
    X_test.to_parquet(os.path.join(PROCESSED_DATA_DIR, f"test_data_{date}.parquet"))
    y_train.to_frame().to_parquet(os.path.join(PROCESSED_DATA_DIR, f"train_labels_{date}.parquet"))
    y_test.to_frame().to_parquet(os.path.join(PROCESSED_DATA_DIR, f"test_labels_{date}.parquet"))

    # 5.3 Initializing Preprocessing Pipeline
    preprocessor = build_preprocessing_pipeline(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

    # 5.4 Fit and Transform Training Data
    logging.info("Fitting and transforming training data...")
    X_train_preprocessed = (preprocessor.fit_transform(X_train)).toarray()
    logging.info(f"✅ Training data fit & transformed. Shape {X_train_preprocessed.shape}")
    X_test_preprocessed = (preprocessor.transform(X_test)).toarray()
    logging.info(f"✅ Testing data transformed. Shape {X_test_preprocessed.shape}")

    feature_names = preprocessor.named_steps['preprocessor'].get_feature_names_out()

    # Convert back to DataFrame
    X_train_preprocessed_df = pd.DataFrame(
        X_train_preprocessed, 
        columns=feature_names 
    )
    X_test_preprocessed_df = pd.DataFrame(
        X_test_preprocessed, 
        columns=feature_names
    )

    # Save Preprocessor and Preprocessed Data
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, f"taxi_fare_preprocessor_{date}.pkl"))

    X_train_preprocessed_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, f"train_preprocessed_{date}.parquet"))
    X_test_preprocessed_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, f"test_preprocessed_{date}.parquet"))

    logging.info("✅ Processing complete! Preprocessor and Data saved and ready for modeling.")


    # 6. Model Training
    logging.info("\n[6] Training model...")
    model = train_model(X_train_preprocessed, y_train)
    model_path = os.path.join(MODELS_DIR, f"taxi_fare_model_{date}.pkl")
    joblib.dump(model, model_path)
    logging.info(f"✅ Model trained and saved to {model_path}")


    # 7. Model Evaluation
    logging.info("\n[7] Evaluating model...")
    metrics = evaluate_model(model, X_test_preprocessed, y_test)

    # Save Metrics
    metrics_path = os.path.join(REPORTS_DIR, f"model_metrics_{date}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"✅ Model evaluated. Metrics saved to {metrics_path}")
    logging.info("\n🏁 Full pipeline execution complete!")

if __name__ == "__main__":
    main()
