import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# Define file paths
DATA_PATH = os.path.join("data/leak_predictions_results.csv")
TEST_PATH = os.path.join("..", "data", "test_leak_cases.csv")
MODEL_PATH = os.path.join("..", "models", "leak_detection_model.pkl")
SCALER_PATH = os.path.join("..", "models", "scaler.pkl")
RESULTS_PATH = os.path.join("..", "data", "leak_predictions_results.csv")

### Step 2: Load Dataset
df = pd.read_csv(DATA_PATH)
print("✅ Dataset Loaded Successfully!")
print(df.head())

### Step 3: Data Preprocessing
df.dropna(inplace=True)  # Remove missing values

# Convert categorical features
df = pd.get_dummies(df, columns=['Pipe Section'], drop_first=True)

# Drop unnecessary columns
df.drop(columns=['Timestamp', 'Sensor ID'], inplace=True, errors='ignore')

### Step 4: Define Features & Target
X = df.drop(columns=['Leak Status'])
y = df['Leak Status']

# Ensure feature consistency
FEATURES = X.columns  # Save feature order for Flask API compatibility

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Step 5: Train Machine Learning Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("✅ Model Training Completed!")

### Step 6: Evaluate Model
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

### Step 7: Save Trained Model & Scaler
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# Save feature order for Flask API
with open("../models/features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

print("✅ Model, Scaler, and Features Saved Successfully!")

### Step 8: Load Test Cases & Predict
test_df = pd.read_csv(TEST_PATH)
test_df.drop(columns=["Leak Status"], errors='ignore', inplace=True)

# Ensure test features match trained model features
missing_cols = set(FEATURES) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0  # Add missing columns with default 0

# Reorder columns
test_df = test_df[FEATURES]

# Load trained model & scaler
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Scale test data & predict
test_df_scaled = scaler.transform(test_df)
test_df['Leak_Predicted'] = model.predict(test_df_scaled)

# Save predictions
test_df.to_csv(RESULTS_PATH, index=False)
print("✅ Predictions Saved in 'leak_predictions_results.csv'")
