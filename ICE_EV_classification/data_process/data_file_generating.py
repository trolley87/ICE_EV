"""
Created by: Jun Zhao @ UofA
Date: 9/29/24
Description: For questions or issues, please refer to zhaojun@arizona.edu.

"""

import os
import shutil

# Define the root data folder and destination folder
root_folder = '/Users/junzhao/Documents/UArizona/papers/EV/data/2_Vehicle_Higher_Desired_Speed_Raw_Processed'  # Replace with the actual path
destination_folder = '/Users/junzhao/Documents/UArizona/papers/EV/data/#ACC/ICE/2_Vehicle_higher/'

# Dictionary to map gap settings from short forms to full names
gap_mapping = {
    'L': 'Long',
    'M': 'Medium',
    'S': 'Short'
}

# Get all .csv files from the root folder (limit to 10 for testing)
csv_files = [f for f in os.listdir(root_folder) if f.endswith('.csv')]
print(csv_files)

# Loop through each file and process it
for file in csv_files:
    # Split the file name into components using the underscore separator
    file_name_parts = file.replace('.csv', '').split('_')

    # Parse the components from the file name
    free_flow_speed = file_name_parts[0]
    speed_fluctuation = file_name_parts[1]
    desired_speed = 0
    gap_setting = file_name_parts[2]
    data_index = file_name_parts[3]

    # Map the gap setting to the full name (Long, Medium, Short)
    gap_full_name = gap_mapping.get(gap_setting, gap_setting)

    # Build the new subfolder path
    new_folder_path = os.path.join(
        destination_folder,
        gap_full_name,
        f"{desired_speed}_desired",
        free_flow_speed,
        speed_fluctuation
    )

    # Ensure the directory structure exists
    os.makedirs(new_folder_path, exist_ok=True)

    # Construct the source file path (original location) and destination file path
    source_file_path = os.path.join(root_folder, file)
    destination_file_path = os.path.join(new_folder_path, f"{data_index}.csv")

    # Move (or copy) the file to the new destination
    shutil.copy(source_file_path, destination_file_path)

    # Print out the file name and its new destination for verification
    print(f"File '{file}' transferred to '{destination_file_path}'")

print("\nFile transfer completed successfully.")