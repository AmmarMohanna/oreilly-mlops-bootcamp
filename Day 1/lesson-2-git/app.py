import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import joblib

# Load Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names) # this is the features of the dataset
df['Target'] = data.target  # Class labels (0: Setosa, 1: Versicolor, 2: Virginica)

# Split into features and labels
X = df.drop(columns=['Target'])
y = df['Target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Scale the data (optional, improves performance for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "iris_model.pkl")
print("Model saved as 'iris_model.pkl'")


