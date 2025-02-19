import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
import pandas as pd
import math

def plot_numerical_histograms(df, numerical_features):
    """
    Plots histograms for numerical features in a panel plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing numerical features.
    - numerical_features (list): List of numerical feature column names.

    Returns:
    - None (Displays histogram plots)
    """
    # Set up the figure and axes for the panel plot
    fig, axes = plt.subplots(nrows=1, ncols=len(numerical_features), figsize=(5 * len(numerical_features), 5))

    # If only one numerical feature, make axes iterable
    if len(numerical_features) == 1:
        axes = [axes]

    # Plot histograms for each numerical feature
    for ax, feature in zip(axes, numerical_features):
        df[feature].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f'Histogram of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')

    # Adjust layout for better visibility
    plt.tight_layout()
    plt.show()


def plot_categorical_bars(df, categorical_features, max_cols=5):
    """
    Plots bar charts for categorical features in a panel plot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing categorical features.
    - categorical_features (list): List of categorical feature column names.
    - max_cols (int): Maximum number of columns in the subplot grid (default: 5).

    Returns:
    - None (Displays bar plots)
    """
    num_features = len(categorical_features)
    
    # Determine number of rows and columns
    num_rows = math.ceil(num_features / max_cols)  # Round up to ensure all features fit
    num_cols = min(num_features, max_cols)  # Limit columns to max_cols or the number of features

    # Set up the figure and axes
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 5 * num_rows))
    
    # Flatten axes array if there are multiple rows
    axes = axes.flatten() if num_features > 1 else [axes]

    # Plot bar charts for each categorical feature
    for ax, feature in zip(axes, categorical_features):
        value_counts = df[feature].value_counts()  # Get category counts
        value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'Bar Chart of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotate labels for readability

    # Hide unused subplots if necessary
    for i in range(len(categorical_features), len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_numerical_boxplots(df, numerical_features, max_cols=4, log_transform=False):
    """
    Plots boxplots for numerical features to detect outliers.
    Optionally applies log transformation while keeping original x-axis labels.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing numerical features.
    - numerical_features (list): List of numerical feature column names.
    - max_cols (int): Maximum number of columns per row in the subplot grid.
    - log_transform (bool): If True, applies log transformation to the data for visualization,
                            but keeps the x-axis in original scale.

    Returns:
    - None (Displays boxplots)
    """
    num_features = len(numerical_features)
    num_rows = -(-num_features // max_cols)  # Compute number of rows (ceil division)

    fig, axes = plt.subplots(num_rows, min(num_features, max_cols), figsize=(5 * min(num_features, max_cols), 5 * num_rows))

    # If only one numerical feature, ensure axes is iterable
    if num_features == 1:
        axes = [[axes]]

    axes = axes.flatten() if num_features > 1 else [axes]

    for ax, feature in zip(axes, numerical_features):
        data = df[feature].copy()

        # Apply log transformation if enabled
        if log_transform:
            data_transformed = np.log1p(data)  # log(1 + x) transformation
        else:
            data_transformed = data

        # Create boxplot
        sns.boxplot(x=data_transformed, ax=ax, color='skyblue')

        # Title and labels
        ax.set_title(f'Boxplot of {feature}' + (' (Log Transformed)' if log_transform else ''))
        ax.set_xlabel(feature)

        # Restore original x-axis labels
        if log_transform:
            ax.set_xticks(ax.get_xticks())  # Keep the same tick positions
            ax.set_xticklabels([f"{val:.2f}" for val in np.exp(ax.get_xticks()) - 1])  # Convert back to original scale

    # Remove empty subplots
    for i in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_trip_outliers(df):
    """
    Filters and plots trip distance vs. trip duration for trips above the IQR.

    Parameters:
    - df (pd.DataFrame): Taxi trip DataFrame with 'trip_distance' and 'trip_duration_minutes'.

    Returns:
    - None (Displays scatter plot)
    """
    if 'trip_distance' not in df.columns or 'trip_duration_minutes' not in df.columns:
        raise ValueError("DataFrame must contain 'trip_distance' and 'trip_duration_minutes' columns.")

    # Compute IQR for trip duration and trip distance
    Q1_duration = df['trip_duration_minutes'].quantile(0.25)
    Q3_duration = df['trip_duration_minutes'].quantile(0.75)
    IQR_duration = Q3_duration - Q1_duration

    Q1_distance = df['trip_distance'].quantile(0.25)
    Q3_distance = df['trip_distance'].quantile(0.75)
    IQR_distance = Q3_distance - Q1_distance

    # Filter trips where either trip distance or duration is above the IQR threshold
    df_outliers = df[
        (df['trip_duration_minutes'] > (Q3_duration + 1.5 * IQR_duration)) |
        (df['trip_distance'] > (Q3_distance + 1.5 * IQR_distance))
    ]

    print(f"Filtered {df_outliers.shape[0]} trips above the IQR threshold.")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_outliers, x='trip_duration_minutes', y='trip_distance', alpha=0.5, edgecolor=None
    )
    
    plt.title("Trip Distance vs. Trip Duration (Outliers Above IQR)")
    plt.xlabel("Trip Duration (minutes)")
    plt.ylabel("Trip Distance (miles)")
    plt.grid(True)
    plt.show()


def plot_trip_outliers_3d_interactive(df):
    """
    Filters and plots an interactive 3D scatter plot of trip distance vs. trip duration vs. estimated speed
    for trips above the IQR using Plotly.

    Parameters:
    - df (pd.DataFrame): Taxi trip DataFrame with 'trip_distance', 'trip_duration_minutes', and 'est_avg_mph'.

    Returns:
    - None (Displays interactive 3D scatter plot)
    """
    if not {'trip_distance', 'trip_duration_minutes', 'est_avg_mph'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'trip_distance', 'trip_duration_minutes', and 'est_avg_mph' columns.")

    # Compute IQR for trip duration and trip distance
    Q1_duration = df['trip_duration_minutes'].quantile(0.25)
    Q3_duration = df['trip_duration_minutes'].quantile(0.75)
    IQR_duration = Q3_duration - Q1_duration

    Q1_distance = df['trip_distance'].quantile(0.25)
    Q3_distance = df['trip_distance'].quantile(0.75)
    IQR_distance = Q3_distance - Q1_distance

    # Filter trips where either trip distance or duration is above the IQR threshold
    df_outliers = df[
        (df['trip_duration_minutes'] > (Q3_duration + 1.5 * IQR_duration)) |
        (df['trip_distance'] > (Q3_distance + 1.5 * IQR_distance))
    ]

    print(f"Filtered {df_outliers.shape[0]} trips above the IQR threshold.")

    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        df_outliers,
        x='trip_duration_minutes',
        y='trip_distance',
        z='est_avg_mph',
        color='est_avg_mph',  # Color by estimated speed
        color_continuous_scale='Viridis',
        title="Trip Distance vs. Trip Duration vs. Estimated Speed (Outliers Above IQR)",
        labels={'trip_duration_minutes': 'Trip Duration (minutes)',
                'trip_distance': 'Trip Distance (miles)',
                'est_avg_mph': 'Estimated Speed (mph)'}
    )

    fig.show(renderer="browser")  # Open in a new browser tab

