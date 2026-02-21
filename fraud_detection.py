# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import pyod
from pyod.models.auto_encoder import AutoEncoder
import inspect

# Step 1: Load Dataset
# Load the Kaggle dataset (update path if needed)
data = pd.read_csv("/content/creditcard.csv")

print("Dataset shape:", data.shape)
print(data.head())

# Step 2: Data Preprocessing
# Separate features and labels
X = data.drop("Class", axis=1)
y = data["Class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 3: Build AutoEncoder Model
# Initialize AutoEncoder model
model = AutoEncoder(
    hidden_neuron_list=[64, 32, 32, 64], # Corrected parameter name based on signature
    epoch_num=30, # Corrected parameter name from 'epochs' to 'epoch_num'
    batch_size=256,
    contamination=0.017
)

# Train model
model.fit(X_train)

# Step 4: Predictions
# Predict labels
y_pred = model.predict(X_test)

# Predict anomaly scores
y_scores = model.decision_function(X_test)

# Step 5: Model Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_scores))

# Step 6: Visualization
plt.figure(figsize=(6, 4))
sns.histplot(y_scores, bins=50)
plt.title("Reconstruction Error Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.show()