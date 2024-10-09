"""
Created by: Jun Zhao @ UofA
Date: 9/29/24
Description: For questions or issues, please refer to zhaojun@arizona.edu.

"""
import os
import pandas as pd


def process_csv(file_path, new_file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Ensure required columns exist before processing
    required_columns = ['Time', 'Smooth Speed Follower', 'Smooth Speed Leader', 'Spacing']
    if all(col in df.columns for col in required_columns):
        # Reset the 'Time' column so that it starts from 0
        df['Time'] = df['Time'] - df['Time'].iloc[0]

        df_resampled = df.groupby(df.index // 10).mean().reset_index(drop=True)

        # Calculate 'Smoothed Acceleration Follower' and 'Smoothed Acceleration Leader'
        df_resampled['Smoothed Acceleration Follower'] = df_resampled['Smooth Speed Follower'].diff() / df_resampled[
            'Time'].diff()
        df_resampled['Smoothed Acceleration Leader'] = df_resampled['Smooth Speed Leader'].diff() / df_resampled['Time'].diff()

        # Calculate the speed difference between follower and leader
        df_resampled['Smooth Speed Difference'] = df_resampled['Smooth Speed Leader'] - df_resampled['Smooth Speed Follower']

        # Save the processed data to the new folder
        df_resampled.to_csv(new_file_path, index=False)
        print(f"Processed and saved file: {new_file_path}")


def process_all_files(root_folder, new_root_folder):
    # Walk through all the files in the root folder and its subfolders
    for subdir, _, files in os.walk(root_folder):
        if '0_desired' in subdir and '10_desired' not in subdir:  # Check if '0_desired' is in the folder path
            for file in files:
                if file.endswith('.csv'):  # Only process .csv files
                    file_path = os.path.join(subdir, file)  # Original file path

                    # Generate the corresponding path in the new folder
                    relative_path = os.path.relpath(subdir, root_folder)  # Relative subfolder path
                    new_subdir = os.path.join(new_root_folder, relative_path)  # Equivalent subfolder in new folder

                    # Ensure the subfolder exists in the new directory
                    if not os.path.exists(new_subdir):
                        os.makedirs(new_subdir)

                    # Full path for the new file
                    new_file_path = os.path.join(new_subdir, file)

                    # Process the file and save it in the new location
                    process_csv(file_path, new_file_path)


# Input folder with original .csv files
root_folder = '/Users/junzhao/Documents/UArizona/papers/EV/data/#ACC/EV/Polestar'

# Output folder where processed files with acceleration data will be saved
new_root_folder = '/Users/junzhao/Documents/UArizona/papers/EV/data/#ACC/EV/Polestar_with_acceleration'

# Start processing all files
process_all_files(root_folder, new_root_folder)