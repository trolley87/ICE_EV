"""
Created by: Jun Zhao @ UofA
Date: 10/7/24
Description: For questions or issues, please refer to zhaojun@arizona.edu.

"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
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

# Function to collect features from all files in the given folders (EV and ICE)
def collect_all_data(ev_folders, ice_folders):
    data = []
    labels = []

    # Collect data from all EV folders
    for ev_folder in ev_folders:
        for subdir, _, files in os.walk(ev_folder):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir, file)
                    ev_features = extract_features(file_path)
                    if ev_features:
                        data.append(ev_features)
                        labels.append(1)  # EV label

    # Collect data from all ICE folders
    for ice_folder in ice_folders:
        for subdir, _, files in os.walk(ice_folder):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(subdir, file)
                    ice_features = extract_features(file_path)
                    if ice_features:
                        data.append(ice_features)
                        labels.append(0)  # ICE label

    return pd.DataFrame(data), labels

# Collect all data for training
df_features, labels = collect_all_data(ev_folders, ice_folders)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_features, labels, test_size=0.2, random_state=42)

# Define classifiers to evaluate
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Function to train and evaluate models
def train_and_evaluate_model(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Evaluate all classifiers and print their accuracies
results = {}
for name, clf in classifiers.items():
    accuracy = train_and_evaluate_model(clf, X_train, y_train, X_test, y_test)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

# Initialize and train the Random Forest classifier for plotting
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

# Get feature importance from the Random Forest classifier
importances = clf_rf.feature_importances_
features = df_features.columns

# Sort features by importance and get the two most important features
sorted_indices = np.argsort(importances)[::-1]
top_two_features = features[sorted_indices[:2]]

# Print the two most important features
print(f"The two most important features are: {top_two_features[0]} and {top_two_features[1]}")

# Predict on the test set
y_pred = clf_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Plotting the classification results using the top two features
plt.figure(figsize=(10, 6))

# Create a scatter plot
sns.scatterplot(
    x=X_test[top_two_features[0]],
    y=X_test[top_two_features[1]],
    hue=y_test,  # Color by actual labels (ICE=0, EV=1)
    style=(y_test == y_pred),  # Marker style based on prediction accuracy
    palette={0: "green", 1: "blue"},  # Colors: green for ICE, blue for EV
    markers={True: "o", False: "X"},  # Circle for correct, X for incorrect predictions
    s=100  # Size of the markers
)

# Set plot labels and title
plt.title(f'Random Forest Classification Results (Accuracy: {accuracy * 100:.2f}%)')
plt.xlabel(top_two_features[0])
plt.ylabel(top_two_features[1])

# Create custom legend
correct_patch = plt.Line2D([0], [0], marker='o', color='w', label='Correct Predictions',
                            markerfacecolor='black', markersize=10)
incorrect_patch = plt.Line2D([0], [0], marker='X', color='w', label='Incorrect Predictions',
                              markerfacecolor='black', markersize=10)

# Add the legend for actual classes
plt.legend(handles=[correct_patch, incorrect_patch], title="Prediction Status", loc='upper right')
plt.scatter([], [], c='green', label='ICE', s=100, marker='o')  # Placeholder for ICE
plt.scatter([], [], c='blue', label='EV', s=100, marker='o')  # Placeholder for EV
plt.legend(title="Actual Class", loc='upper right', fontsize='medium')

# Display the grid and plot
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# After making predictions with the Random Forest classifier
y_pred = clf_rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Calculate and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ICE", "EV"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
