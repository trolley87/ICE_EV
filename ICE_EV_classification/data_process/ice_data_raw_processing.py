"""
Created by: Jun Zhao @ UofA
Date: 9/29/24
Description: For questions or issues, please refer to zhaojun@arizona.edu.

"""
import pandas as pd
import os


import pandas as pd
import os

def process_vehicle_data(file_name):
    # Input and output folder paths
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Full file paths
    input_file_path = os.path.join(input_folder, file_name)
    output_file_path = os.path.join(output_folder, file_name)

    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Convert the 'timestamps' column to datetime, allowing mixed formats
    if 'timestamps' in df.columns:
        df['timestamps'] = pd.to_datetime(df['timestamps'], format='mixed', errors='coerce')

        # Drop any rows where timestamp parsing failed (if any)
        df = df.dropna(subset=['timestamps'])

        # Calculate the time delta from the first timestamp (set the first timestamp as time 0)
        df['Time_delta'] = (df['timestamps'] - df['timestamps'].iloc[0]).dt.total_seconds().round(2)

    # Define the required columns
    required_columns = ['Time_delta', 'speed1', 'speed2', 'Smoothed_speed1', 'Smoothed_speed2',
                        'Difference', 'Inital time gap']

    # Check if all required columns are present
    if not all(column in df.columns for column in required_columns):
        print(f"Skipping file {file_name} as it lacks one or more required columns.")
        return  # Skip processing this file

    # Convert the 'timestamps' column to datetime, allowing mixed formats
    if 'timestamps' in df.columns:
        df['timestamps'] = pd.to_datetime(df['timestamps'], format='mixed', errors='coerce')

        # Drop any rows where timestamp parsing failed (if any)
        df = df.dropna(subset=['timestamps'])

        # Calculate the time delta from the first timestamp (set the first timestamp as time 0)
        df['Time_delta'] = (df['timestamps'] - df['timestamps'].iloc[0]).dt.total_seconds().round(2)


    # Check which columns are present in the DataFrame
    existing_columns = [col for col in required_columns if col in df.columns]

    # Create a mapping for renaming the existing columns
    column_mapping = {
        'Time_delta': 'Time',
        'speed1': 'Speed Leader',
        'speed2': 'Speed Follower',
        'Smoothed_speed1': 'Smooth Speed Leader',
        'Smoothed_speed2': 'Smooth Speed Follower'
    }

    # Filter the DataFrame to keep only the existing columns
    df_processed = df[existing_columns].copy()

    # Rename the columns that exist
    df_processed.rename(columns=column_mapping, inplace=True)

    # Calculate the initial distance gap using the first row's values
    initial_distance_gap = (df_processed['Inital time gap'].iloc[0] * df_processed['Speed Leader'].iloc[0]) / 3.6

    # Calculate spacing as initial distance gap + difference
    df_processed['Spacing'] = initial_distance_gap + df_processed['Difference']

    # Recalculate smoothed accelerations
    df_processed['Smoothed Acceleration Leader'] = df_processed['Smooth Speed Leader'].diff() / df_processed['Time'].diff()
    df_processed['Smoothed Acceleration Follower'] = df_processed['Smooth Speed Follower'].diff() / df_processed['Time'].diff()

    # Add the speed difference column (Speed Follower - Speed Leader)
    df_processed['Smooth Speed Difference'] = df_processed['Smooth Speed Follower'] - df_processed['Smooth Speed Leader']

    df_processed['Speed Difference'] = df_processed['Speed Leader'] - df_processed['Speed Follower']


    # Save the processed data to the new folder with the same file name
    df_processed.to_csv(output_file_path, index=False)

    print(f"Processed file saved to: {output_file_path}")


def process_files(input_folder):
    # Get the list of all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    for file_name in csv_files:
        print(f"Processing file: {file_name}")
        process_vehicle_data(file_name)


input_folder = '/Users/junzhao/Documents/UArizona/papers/EV/data/2_Vehicle_Same_Desired_Speed_Raw/'
output_folder = '/Users/junzhao/Documents/UArizona/papers/EV/data/2_Vehicle_Same_Desired_Speed_Raw_Processed/'


# Call the function to process the top 3 files
process_files(input_folder)