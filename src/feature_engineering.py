import os
import pandas as pd
import numpy as np
import holidays
import requests
import re
import logging
from datetime import datetime, timedelta

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def add_trip_datetime_features(df, drop_original=False):
    """
    Computes trip duration in minutes and extracts time-related features.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'tpep_pickup_datetime' and 'tpep_dropoff_datetime' columns.
    drop_original (bool): If True, drops the original datetime columns after processing.

    Returns:
    pd.DataFrame: DataFrame with new trip duration and time-based features.
    """

    # Check if both datetime columns exist in the DataFrame
    required_columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logging.error(f"❌ Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert to datetime if not already
    for col in required_columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):  
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Ensure there are no NaT values before computing differences
    if df[required_columns].isna().any().any():
        logging.error("❌ Some datetime values could not be converted. Check for missing or invalid data.")
        raise ValueError("Some datetime values could not be converted. Check for missing or invalid data.")


    # Ensure datetime columns are in correct format
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Compute trip duration in minutes
    df['trip_duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Extract useful time features
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour  # Hour of the day (0-23)
    df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek  # Monday=0, Sunday=6

    # Optional features for modeling where we have data for multiple months/years
    # df['pickup_month'] = df['tpep_pickup_datetime'].dt.month  # Month of the year (1-12)
    # df['pickup_day'] = df['tpep_pickup_datetime'].dt.day  # Day of the month (1-31)
    # df['pickup_year'] = df['tpep_pickup_datetime'].dt.year  # Year

    # Check if pickup date is a holiday
    us_holidays = holidays.US()
    df['is_holiday'] = df['tpep_pickup_datetime'].apply(lambda x: 1 if x.date() in us_holidays else 0)

    # Drop original datetime columns if specified
    if drop_original:
        df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)

    return df


def add_rain_or_snow_column(trips_df, weather_df):
    """
    Adds a 'rain_or_snow' column to the ride data based on weather conditions.

    Mapping:
    - 0 = No precipitation
    - 1 = Rain only
    - 2 = Snow only
    - 3 = Both rain and snow

    Parameters:
    - trips_df (pd.DataFrame): Taxi trip data with 'tpep_pickup_datetime'.
    - weather_df (pd.DataFrame): Weather data with 'datetime' and 'preciptype' columns.

    Returns:
    - pd.DataFrame: Taxi trip DataFrame with an added 'rain_or_snow' column.
    """
    # Ensure datetime columns are parsed correctly and have the same precision
    trips_df = trips_df.copy()
    trips_df['tpep_pickup_datetime'] = pd.to_datetime(trips_df['tpep_pickup_datetime']).astype('datetime64[ns]')  # Convert to seconds precision
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).astype('datetime64[ns]')  # Convert to seconds precision

    # Process 'preciptype' column in weather_df (convert lists to categorical values)
    def categorize_preciptype(precip_list):
        if isinstance(precip_list, str):  # Convert string representation of list to actual list
            try:
                precip_list = eval(precip_list)  # Be cautious with eval()
            except:
                return 0  # If eval fails, assume no precipitation
        if not precip_list or precip_list is np.nan:
            return 0  # No precipitation
        elif 'rain' in precip_list and 'snow' in precip_list:
            return 3  # Both rain and snow
        elif 'snow' in precip_list:
            return 2  # Only snow
        elif 'rain' in precip_list:
            return 1  # Only rain
        return 0  # Default case (shouldn't be reached)

    # Apply function to categorize precipitation
    weather_df['rain_or_snow'] = weather_df['preciptype'].apply(categorize_preciptype)

    # Drop the original 'preciptype' column after processing
    weather_df = weather_df[['datetime', 'rain_or_snow']]

    # Sort both DataFrames by datetime for merging
    trips_df = trips_df.sort_values('tpep_pickup_datetime')
    weather_df = weather_df.sort_values('datetime')

    # Merge using nearest available weather timestamp
    trips_df = pd.merge_asof(
        trips_df, weather_df,
        left_on='tpep_pickup_datetime', right_on='datetime',
        direction='backward'  # Finds the closest weather report before or at pickup time
    )

    # Drop the merged 'datetime' column after merge
    trips_df.drop(columns=['datetime'], inplace=True)

    return trips_df