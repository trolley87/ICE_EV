import os
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import matplotlib.pyplot as plt

# Define paths for EV and ICE data
ev_folder = '../#ACC/EV/Hyundai_Ioniq_5_with_acceleration'
ice_folder = '../#ACC/ICE/2_Vehicle'

def extract_speed_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check if 'Smooth Speed Follower' and 'Smooth Speed Leader' exist
    if 'Smooth Speed Follower' in df.columns and 'Smooth Speed Leader' in df.columns:
        return df['Smooth Speed Follower'].values, df['Smooth Speed Leader'].values
    return None, None


def find_top_similar_speeds(ev_folder, ice_folder, top_n=10):
    ev_speeds = []
    ice_speeds = []

    # Collect EV smoothed follower and leader speed profiles
    for subdir, _, files in os.walk(ev_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                smoothed_follower_speed, smoothed_leader_speed = extract_speed_data(file_path)
                if smoothed_follower_speed is not None and smoothed_leader_speed is not None:
                    ev_speeds.append((file_path, file, smoothed_follower_speed, smoothed_leader_speed))

    # Collect ICE smoothed follower and leader speed profiles
    for subdir, _, files in os.walk(ice_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                smoothed_follower_speed, smoothed_leader_speed = extract_speed_data(file_path)
                if smoothed_follower_speed is not None and smoothed_leader_speed is not None:
                    ice_speeds.append((file_path, file, smoothed_follower_speed, smoothed_leader_speed))

    # List to store all matches with distances
    matches = []

    # Compare EV profiles with ICE profiles and find top_n most similar pairs
    for ev_path, ev_file, ev_follower_speed, ev_leader_speed in ev_speeds:
        for ice_path, ice_file, ice_follower_speed, ice_leader_speed in ice_speeds:
            # Ensure the speeds have the same length, trim the longer one if necessary
            min_length = min(len(ev_leader_speed), len(ice_leader_speed))
            ev_leader_trimmed = ev_leader_speed[:min_length]
            ice_leader_trimmed = ice_leader_speed[:min_length]
            ev_follower_trimmed = ev_follower_speed[:min_length]
            ice_follower_trimmed = ice_follower_speed[:min_length]

            # Remove any NaN or inf values from the trimmed arrays
            ev_leader_trimmed = np.nan_to_num(ev_leader_trimmed, nan=0.0, posinf=0.0, neginf=0.0)
            ice_leader_trimmed = np.nan_to_num(ice_leader_trimmed, nan=0.0, posinf=0.0, neginf=0.0)

            # Calculate Euclidean distance for smoothed leader speeds
            distance = euclidean(ev_leader_trimmed, ice_leader_trimmed)

            # Store the result
            matches.append({
                'ev_file': ev_file,
                'ev_path': ev_path,
                'ev_follower_speed': ev_follower_trimmed,
                'ev_leader_speed': ev_leader_trimmed,
                'ice_file': ice_file,
                'ice_path': ice_path,
                'ice_follower_speed': ice_follower_trimmed,
                'ice_leader_speed': ice_leader_trimmed,
                'distance': distance
            })

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x['distance'])

    # Return the top N matches
    return matches[:top_n]


# Find the top 10 most similar speeds
top_matches = find_top_similar_speeds(ev_folder, ice_folder, top_n=10)

# Plot the top 10 most similar smoothed follower and leader speeds
for i, match in enumerate(top_matches):
    plt.figure(figsize=(10, 6))
    time = range(len(match['ev_follower_speed']))  # Assuming equal time intervals

    # Plot EV and ICE smoothed follower speeds
    plt.plot(time, match['ev_follower_speed'], label=f'EV Smoothed Follower Speed ({match["ev_file"]})', color='blue', linestyle='dashed')
    plt.plot(time, match['ice_follower_speed'], label=f'ICE Smoothed Follower Speed ({match["ice_file"]})', color='green', linestyle='dashed')

    # Plot EV and ICE smoothed leader speeds
    plt.plot(time, match['ev_leader_speed'], label=f'EV Smoothed Leader Speed ({match["ev_file"]})', color='blue')
    plt.plot(time, match['ice_leader_speed'], label=f'ICE Smoothed Leader Speed ({match["ice_file"]})', color='green')

    # Set plot title and labels, including folder names and files
    plt.title(f'Top {i+1}: Smoothed Follower and Leader Speeds\nEV: {match["ev_path"]}\nICE: {match["ice_path"]}\n(Distance: {match["distance"]:.4f})')
    plt.xlabel('Time')
    plt.ylabel('Speed (km/h)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()
