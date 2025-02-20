import os
from dotenv import load_dotenv

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory containing raw data files
DATA_NAME = 'yellow_tripdata_2024-01.parquet'
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
TAXI_DATA_PATH = os.path.join(RAW_DATA_DIR, DATA_NAME)

# Directory for cleaned data files
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# path to .env file in the project root directory
DOTENV_PATH = os.path.join(BASE_DIR, ".env")

# API info and directory for weather data
if os.path.exists(DOTENV_PATH):
    load_dotenv(DOTENV_PATH)
else:
    raise FileNotFoundError("No .env file found. Please initialize required folders and environment using the initialize_project.py script.")

API_KEY = os.getenv("API_KEY")

if not API_KEY or API_KEY == "None":
    raise ValueError("API_KEY not found in .env file. Check the README for instructions on setting up the Visual Crossing Weather API.")

WEATHER_DIR = os.path.join(BASE_DIR, "data", "weather")

# Model directory
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Reports directory
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Feature Categories
DROP_COLUMNS = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 'Airport_fee', 'est_avg_mph', 'trip_duration_hours']
NUMERICAL_FEATURES = ['passenger_count', 'trip_distance', 'trip_duration_minutes']
CATEGORICAL_FEATURES = ['VendorID', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'payment_type', 'pickup_day_of_week', 'pickup_hour', 'is_holiday', 'rain_or_snow']
TARGET_FEATURE = 'total_amount'

# TRAIN-TEST SPLIT RANDOM STATE
RANDOM_STATE = 42

# Thresholds for Data Cleaning
MAX_TRIP_DURATION_HOURS = 12 * 60 # 12 hours in minutes
MIN_TRIP_DURATION_MINUTES = 1 # remove trips below 1 minute duration
MAX_ESTIMATED_SPEED_MPH = 90 # remove unrealistically high speeds
MIN_ESTIMATED_SPEED_MPH = 1 # remove unrealistically low speeds

# Joblib backend directory (used in model_training.py)
JOBLIB_BACKEND_DIR = "G:/" # Change this to a directory with sufficient space on your machine if needed

# Model and Preprocessor paths (INFERENCE)
MODEL_PATH = "models/taxi_fare_model_2024-01.pkl"
PREPROCESSOR_PATH = "models/taxi_fare_preprocessor_2024-01.pkl"