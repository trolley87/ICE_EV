"""
Created by: Jun Zhao @ UofA
Date: 9/29/24
Description: For questions or issues, please refer to zhaojun@arizona.edu.

"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
import numpy as np

# Define paths for multiple EV folders and ICE folders
ev_folders = [
    '../#ACC/EV/Hyundai_Ioniq_5_with_acceleration',
    '../#ACC/EV/Polestar_with_acceleration',
    '../#ACC/EV/Tesla_with_acceleration'
]

ice_folders = [
    '../#ACC/ICE/2_Vehicle',
    '../#ACC/ICE/2_Vehicle_higher'
]

# Function to extract features related to the follower vehicle only
def extract_features(file_path):
    df = pd.read_csv(file_path)
    if 'Speed Follower' in df.columns and 'Smoothed Acceleration Follower' in df.columns and 'Smooth Speed Difference' in df.columns:
        return {
            'mean_speed_follower': df['Speed Follower'].mean(),
            'std_speed_follower': df['Speed Follower'].std(),
            'mean_acceleration_follower': df['Smoothed Acceleration Follower'].mean(),
            'std_acceleration_follower': df['Smoothed Acceleration Follower'].std(),
            'mean_spacing': df['Spacing'].mean(),
            'std_spacing': df['Spacing'].std(),
            'mean_smooth_speed_difference': df['Smooth Speed Difference'].mean(),
            'std_smooth_speed_difference': df['Smooth Speed Difference'].std()
        }
    return None

def extract_speed_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check if 'Smooth Speed Leader' exists
    if 'Smooth Speed Leader' in df.columns:
        return df['Smooth Speed Leader'].values
    return None

def find_top_100_similar_speeds(ev_folders, ice_folders, top_n=100):
    ev_speeds = []
    ice_speeds = []

    # Collect EV smoothed leader speed profiles from multiple EV folders
    for ev_folder in ev_folders:
        for subdir, _, files in os.walk(ev_folder):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir, file)
                    smoothed_leader_speed = extract_speed_data(file_path)
                    if smoothed_leader_speed is not None:
                        ev_speeds.append((file_path, file, smoothed_leader_speed))

    # Collect ICE smoothed leader speed profiles from multiple ICE folders
    for ice_folder in ice_folders:
        for subdir, _, files in os.walk(ice_folder):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir, file)
                    smoothed_leader_speed = extract_speed_data(file_path)
                    if smoothed_leader_speed is not None:
                        ice_speeds.append((file_path, file, smoothed_leader_speed))

    # List to store all matches with distances
    matches = []

    # Compare EV profiles with ICE profiles and find top_n most similar pairs
    for ev_path, ev_file, ev_leader_speed in ev_speeds:
        for ice_path, ice_file, ice_leader_speed in ice_speeds:
            # Ensure the speeds have the same length, trim the longer one if necessary
            min_length = min(len(ev_leader_speed), len(ice_leader_speed))
            ev_leader_trimmed = ev_leader_speed[:min_length]
            ice_leader_trimmed = ice_leader_speed[:min_length]

            # Remove any NaN or inf values from the trimmed arrays
            ev_leader_trimmed = np.nan_to_num(ev_leader_trimmed, nan=0.0, posinf=0.0, neginf=0.0)
            ice_leader_trimmed = np.nan_to_num(ice_leader_trimmed, nan=0.0, posinf=0.0, neginf=0.0)

            # Calculate Euclidean distance for smoothed leader speeds
            distance = euclidean(ev_leader_trimmed, ice_leader_trimmed)

            # Store the result
            matches.append({
                'ev_path': ev_path,
                'ev_file': ev_file,
                'ice_path': ice_path,
                'ice_file': ice_file,
                'distance': distance
            })

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x['distance'])

    # Return the top N matches
    return matches[:top_n]

# Function to train and evaluate models
def train_and_evaluate_model(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Get the top 100 most similar speeds
top_100_matches = find_top_100_similar_speeds(ev_folders, ice_folders, top_n=5000)

# Data preparation for top 100 matches
data = []
labels = []

for match in top_100_matches:
    ev_features = extract_features(match['ev_path'])
    if ev_features:
        data.append(ev_features)
        labels.append(1)  # EV label

    ice_features = extract_features(match['ice_path'])
    if ice_features:
        data.append(ice_features)
        labels.append(0)  # ICE label

df_features = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df_features, labels, test_size=0.2, random_state=42)

# Classifiers to evaluate
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
}

# Evaluate classifiers
results = {name: train_and_evaluate_model(clf, X_train, y_train, X_test, y_test) for name, clf in classifiers.items()}

# Print results
for name, accuracy in results.items():
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

# Visualization of the Random Forest Results
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize and train the Random Forest classifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

# Feature importance
importances = clf_rf.feature_importances_
features = df_features.columns

# Sorting features by their importance
sorted_indices = np.argsort(importances)[::-1]
top_two_features = features[sorted_indices[:2]]

# Plotting
sns.scatterplot(x=df_features[top_two_features[0]], y=df_features[top_two_features[1]], hue=labels,
                palette={1: "blue", 0: "green"}, style=labels, markers={1: "o", 0: "X"})

plt.title('Random Forest Top Two Features Classification')
plt.xlabel(top_two_features[0])
plt.ylabel(top_two_features[1])
plt.legend(title='Vehicle Type', labels=['ICE', 'EV'])
plt.grid(True)
plt.show()
