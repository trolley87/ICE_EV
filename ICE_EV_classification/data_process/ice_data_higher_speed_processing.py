"""
Created by: Jun Zhao @ UofA
Date: 10/7/24
Description: For questions or issues, please refer to zhaojun@arizona.edu.

"""
import os
import pandas as pd

# Define the smoothing window size
SMOOTHING_WINDOW = 5


# Function to calculate the new columns
def add_new_columns(df):
    # Smooth Speed Leader and Speed Follower using rolling mean (simple moving average)
    df['Smooth Speed Leader'] = df['Speed Leader'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    df['Smooth Speed Follower'] = df['Speed Follower'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    # Calculate Smoothed Acceleration Leader and Smoothed Acceleration Follower (derivative of smoothed speed over time)
    df['Smoothed Acceleration Leader'] = df['Smooth Speed Leader'].diff() / df['Time'].diff()
    df['Smoothed Acceleration Follower'] = df['Smooth Speed Follower'].diff() / df['Time'].diff()

    # Calculate Speed Difference and Smooth Speed Difference
    df['Speed Difference'] = df['Speed Follower'] - df['Speed Leader']
    df['Smooth Speed Difference'] = df['Smooth Speed Follower'] - df['Smooth Speed Leader']

    return df


# Function to process all files and add the new columns
def process_all_files(ice_folder):
    for folder in [ice_folder]:
        for subdir, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir, file)
                    df = pd.read_csv(file_path)

                    # Check if the required columns exist
                    if 'Speed Leader' in df.columns and 'Speed Follower' in df.columns and 'Time' in df.columns:
                        # Add new columns to the dataframe
                        df = add_new_columns(df)

                        # Save the updated dataframe back to the CSV file
                        df.to_csv(file_path, index=False)
                        print(f"Processed file: {file_path}")


# Define paths for multiple EV folders and ICE folder

ice_folder = '/Users/junzhao/Documents/UArizona/papers/EV/data/#ACC/ICE/2_Vehicle_higher'

# Process all files and add the new columns
process_all_files(ice_folder)