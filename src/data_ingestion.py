import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import re
import logging
import boto3
import botocore

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_taxi_data(data_path):
    """
    Loads NYC taxi trip data from a Parquet or CSV file.

    Parameters:
    - data_path (str): Path to the dataset file.

    Returns:
    - pd.DataFrame: Loaded taxi data.
    """
    if not os.path.exists(data_path):
        logging.error(f"❌ Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        logging.error(f"❌ Unsupported file format. Use Parquet or CSV.")
        raise ValueError("Unsupported file format. Use Parquet or CSV.")

    logging.info(f"✅ Taxi data loaded: {df.shape}")
    return df


def extract_month_from_filename(dataset_name):
    """
    Extracts the year and month from a dataset filename in 'YYYY-MM' format.

    Parameters:
    - dataset_name (str): The filename (e.g., 'yellow_tripdata_2024-01.parquet').

    Returns:
    - str: Extracted month string in 'YYYY-MM' format.
    """
    match = re.search(r"(\d{4})-(\d{2})", dataset_name)
    if not match:
        logging.error("❌ Filename does not contain a valid YYYY-MM format.")
        raise ValueError("Filename does not contain a valid YYYY-MM format.")
    return match.group(0)


def fetch_weather_data(dataset_name, api_key, save_path="data/weather/"):
    """
    Fetches hourly weather data for a month and saves it locally. If data already exists, it loads from disk.

    Parameters:
    - dataset_name (str): The filename containing the target month (e.g., 'yellow_tripdata_2024-01.parquet').
    - api_key (str): Your Visual Crossing API key.
    - save_path (str): Directory where weather data should be saved (default: "data/").

    Returns:
    - pd.DataFrame: Weather data for the requested period.
    """
    month_str = extract_month_from_filename(dataset_name)
    weather_file = os.path.join(save_path, f"weather_data_{month_str}.csv")

    # Check if weather data already exists
    if os.path.exists(weather_file):
        logging.info(f"✅ Loading existing weather data from {weather_file}")
        weather_df = pd.read_csv(weather_file)

        try:
            logging.info("🔄 Attempting to reformat weather data...")
            
            # Ensure necessary columns exist before processing
            if 'date' in weather_df.columns and 'datetime' in weather_df.columns:
                weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce')
                weather_df['datetime'] = pd.to_datetime(weather_df['date'].astype(str) + ' ' + weather_df['datetime'].astype(str), errors='coerce')
                weather_df.drop(columns=['date'], inplace=True)
                logging.info("✅ Weather data successfully reformatted.")
            else:
                logging.warning("⚠️ Missing required columns ('date' or 'datetime') in weather data. Skipping reformatting.")

        except Exception as e:
            logging.warning(f"⚠️ Could not reformat weather data due to an issue: {e}")

        return weather_df

    # If file doesn't exist, fetch data from API
    logging.info(f"🌍 Fetching new weather data for {month_str}")

    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    location = "New York, NY, United States"

    # Get start and end dates (buffer 1 day before & after)
    year, month = map(int, month_str.split("-"))
    first_day = datetime(year, month, 1)
    last_day = (first_day.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    start_date = first_day - timedelta(days=1)
    end_date = last_day + timedelta(days=1)

    print(f"Fetching weather data from {start_date.date()} to {end_date.date()}")

    # API URL
    date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    url = f"{base_url}{location}/{date_range}?unitGroup=us&include=hours&elements=datetime,tempmax,tempmin,precip,precipprob,precipcover,preciptype,snow,snowdepth,icon&key={api_key}&contentType=json"

    # Fetch Data
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'days' in data:
            all_hours = []
            for day in data['days']:  # Extract hourly data for each day
                for hour in day.get('hours', []):
                    hour['datetime'] = pd.to_datetime(hour['datetime'])
                    all_hours.append(hour)

            # Convert to DataFrame
            weather_df = pd.DataFrame(all_hours)
            weather_df.to_csv(weather_file, index=False)  # Save to file
            print(f"Successfully fetched & saved {weather_df.shape[0]} hourly records to {weather_file}")
            return weather_df
        else:
            print(f"Warning: No 'days' data found for {date_range}")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
