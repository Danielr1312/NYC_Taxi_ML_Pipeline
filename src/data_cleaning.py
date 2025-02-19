import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timedelta

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def filter_trips_by_dataset_date(df, dataset_name):
    """
    Filters taxi trip data to ensure all trips fall within the dataset's intended month and year,
    allowing a 1-day buffer on both ends and ensuring trips spanning months are not mistakenly removed.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing taxi trip data.
    - dataset_name (str): The filename (e.g., 'yellow_tripdata_2024-01.parquet').

    Returns:
    - pd.DataFrame: A filtered DataFrame containing only valid trips.
    - pd.DataFrame: A DataFrame of trips that are outside the expected range.
    """
    # Extract year and month from the dataset filename
    match = re.search(r'(\d{4})-(\d{2})', dataset_name)
    if not match:
        logging.error("❌ Filename does not contain a valid YYYY-MM format.")
        raise ValueError("Filename does not contain a valid YYYY-MM format.")

    year, month = match.groups()
    start_date = datetime(int(year), int(month), 1) - timedelta(days=1)  # 1-day padding before start
    next_month = datetime(int(year), int(month), 1).replace(day=28) + pd.DateOffset(days=4)
    end_date = (next_month - pd.DateOffset(days=next_month.day)) + timedelta(days=1)  # 1-day padding after end

    logging.info(f"Filtering trips for {year}-{month} (from {start_date.date()} to {end_date.date()}) with a 1-day buffer")

    # Ensure datetime columns are in the correct format
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Keep trips where at least one datetime falls within the padded range
    valid_trips = df[
        (df['tpep_pickup_datetime'] >= start_date) & (df['tpep_pickup_datetime'] <= end_date) |
        (df['tpep_dropoff_datetime'] >= start_date) & (df['tpep_dropoff_datetime'] <= end_date)
    ]

    # Identify removed trips (completely outside the range)
    removed_trips = df.drop(valid_trips.index)

    logging.info(f"Total trips outside {year}-{month}: {removed_trips.shape[0]}")

    return valid_trips, removed_trips


def remove_long_trips(df, max_hours=12):
    """
    Removes trips exceeding the maximum allowed trip duration (default: 12 hours) and returns the removed rows.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'trip_duration_minutes' column.
    - max_hours (int): The maximum allowed trip duration in hours (default is 12).
    
    Returns:
    - pd.DataFrame: Filtered DataFrame without excessively long trips.
    - pd.DataFrame: DataFrame containing the dropped trips.
    """
    max_duration = max_hours * 60  # Convert hours to minutes

    # Filter trips that exceed the max allowed duration
    long_trips = df[df['trip_duration_minutes'] > max_duration]
    df_filtered = df[df['trip_duration_minutes'] <= max_duration]

    logging.info(f"Dropped {long_trips.shape[0]} trips exceeding {max_hours} hours.")

    return df_filtered, long_trips


def remove_short_trips(df, min_minutes=1):
    """
    Removes trips with a duration below a specified threshold (default: 1 minute) 
    and returns the removed trips.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'trip_duration_minutes'.
    - min_minutes (int): The minimum trip duration in minutes (default is 1).

    Returns:
    - pd.DataFrame: Filtered DataFrame without short trips.
    - pd.DataFrame: DataFrame containing the removed short trips.
    """
    short_trips = df[df['trip_duration_minutes'] < min_minutes]
    df_filtered = df[df['trip_duration_minutes'] >= min_minutes]

    logging.info(f"Dropped {short_trips.shape[0]} trips shorter than {min_minutes} minutes.")

    return df_filtered, short_trips


def add_estimated_speed(df):
    """
    Calculates and adds 'est_avg_mph' (estimated average mph) to the dataset.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'trip_distance' and 'trip_duration_minutes'.

    Returns:
    - pd.DataFrame: DataFrame with new 'est_avg_mph' column.
    """
    df = df.copy()  # Prevent modifying the original DataFrame
    
    # Convert trip duration to hours
    df['trip_duration_hours'] = df['trip_duration_minutes'] / 60

    # Compute estimated average speed
    df['est_avg_mph'] = df['trip_distance'] / df['trip_duration_hours']

    # Handle infinite values (caused by zero or near-zero duration)
    df['est_avg_mph'] = df['est_avg_mph'].replace([float('inf'), -float('inf')], np.nan)


    return df


def remove_high_speed_trips(df, max_mph=100):
    """
    Removes trips where the estimated average speed exceeds a specified threshold.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'est_avg_mph'.
    - max_mph (int): The maximum allowed estimated speed (default is 100 mph).

    Returns:
    - pd.DataFrame: Filtered DataFrame without high-speed trips.
    - pd.DataFrame: DataFrame containing the removed high-speed trips.
    """
    high_speed_trips = df[df['est_avg_mph'] > max_mph]
    df_filtered = df[df['est_avg_mph'] <= max_mph]

    logging.info(f"Dropped {high_speed_trips.shape[0]} trips exceeding {max_mph} mph.")

    return df_filtered, high_speed_trips


def remove_low_speed_trips(df, min_mph=1):
    """
    Filters out trips with an estimated average speed below a given threshold.

    Parameters:
    - df (pd.DataFrame): Taxi trip DataFrame containing 'est_avg_mph'.
    - min_mph (float): Minimum reasonable speed (default: 3 mph).

    Returns:
    - pd.DataFrame: Filtered DataFrame without unreasonably slow trips.
    - pd.DataFrame: DataFrame containing removed trips.
    """
    if 'est_avg_mph' not in df.columns:
        logging.error("❌ Filename does not contain a valid YYYY-MM format.")
        raise ValueError("DataFrame must contain 'est_avg_mph' column.")

    # Filter trips with speeds below the threshold
    low_speed_trips = df[df['est_avg_mph'] <= min_mph]
    df_filtered = df[df['est_avg_mph'] > min_mph]

    logging.info(f"Dropped {low_speed_trips.shape[0]} trips with est_avg_mph < {min_mph} mph.")

    return df_filtered, low_speed_trips


def convert_value_to_missing(df, column_name, value_to_replace):
    """
    Converts a specified value in a given column to NaN (missing value).

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - column_name (str): The column in which to replace values.
    - value_to_replace: The value to be converted to NaN.

    Returns:
    - pd.DataFrame: The modified DataFrame with updated missing values.
    """ 
    if column_name not in df.columns:
        logging.error(f"❌ Column '{column_name}' not found in DataFrame.")
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    df = df.copy()
    missing_values_og = df[column_name].isna().sum()
    df[column_name] = df[column_name].replace(value_to_replace, np.nan)

    logging.info(f"Replaced {df[column_name].isna().sum() - missing_values_og} occurrences of '{value_to_replace}' with NaN in '{column_name}'.")

    return df


def remap_unknown_payment_types(df, column_name='payment_type'):
    """
    Remaps payment types outside {1,2,3,4,5,6} to 5 (Unknown).

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the payment type column.
    - column_name (str): The column name for payment types (default: 'payment_type').

    Returns:
    - pd.DataFrame: DataFrame with remapped payment types.
    """
    if column_name not in df.columns:
        logging.error(f"❌ Column '{column_name}' not found in DataFrame.")
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Define valid payment types
    valid_payment_types = {1, 2, 3, 4, 5, 6}

    # Count how many will be remapped
    num_remapped = df[~df[column_name].isin(valid_payment_types)].shape[0]

    # Remap unknown values to 5
    df = df.copy()
    df[column_name] = df[column_name].apply(lambda x: x if x in valid_payment_types else 5)

    logging.info(f"Remapped {num_remapped} payment types to 5 (Unknown).")

    return df


def remove_negative_total_amount(df):
    """
    Filters out trips where total_amount is negative.

    Parameters:
    - df (pd.DataFrame): The taxi trip DataFrame containing 'total_amount'.

    Returns:
    - pd.DataFrame: DataFrame with negative total amounts removed.
    - pd.DataFrame: DataFrame containing removed trips.
    """
    if 'total_amount' not in df.columns:
        logging.error("❌ DataFrame must contain 'total_amount' column.")
        raise ValueError("DataFrame must contain 'total_amount' column.")

    # Filter out negative total amounts
    valid_trips = df[df['total_amount'] >= 0]
    removed_trips = df[df['total_amount'] < 0]  # Store removed rows

    logging.info(f"Removed {removed_trips.shape[0]} trips with negative total amounts.")

    return valid_trips, removed_trips


def remove_unneeded_features(df, columns_to_drop):
    """
    Drops specified columns from the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - columns_to_drop (list): List of column names to remove.

    Returns:
    - pd.DataFrame: DataFrame with specified columns removed.
    """
    if not set(columns_to_drop).issubset(df.columns):
        missing_cols = set(columns_to_drop) - set(df.columns)
        logging.error(f"❌ Columns not found in DataFrame: {missing_cols}")
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    df = df.drop(columns=columns_to_drop)

    logging.info(f"Dropped {len(columns_to_drop)} columns from the DataFrame.")

    return df