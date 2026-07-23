import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle
import os

# Load the dataset
dataset = 'diabetes_prediction_dataset.csv'
df = pd.read_csv(dataset)

# Clean the dataset
df = df.dropna() 

# Encode categorical features
train_df_object = df.select_dtypes(include='object')
encoders = {} 
for col in train_df_object.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features (X) and target (y)
X = df.drop(columns=["diabetes"])
y = df["diabetes"]

# Identify numerical and categorical columns
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
other_features = [col for col in X.columns if col not in numerical_features]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical columns only
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_num_scaled = scaler.transform(X_test[numerical_features])

# Recombine numerical and categorical
X_train_scaled = pd.DataFrame(X_train_num_scaled, columns=numerical_features, index=X_train.index)
X_train_scaled[other_features] = X_train[other_features]

X_test_scaled = pd.DataFrame(X_test_num_scaled, columns=numerical_features, index=X_test.index)
X_test_scaled[other_features] = X_test[other_features]

# Initialize the SMOTE object for class balancing
smt = SMOTE(random_state=42)
X_train_bal, y_train_bal = smt.fit_resample(X_train_scaled, y_train)

# Create models folder if needed
os.makedirs("models", exist_ok=True)

# Train Logistic Regression model
model_a = LogisticRegression(max_iter=1000)
model_a.fit(X_train_bal, y_train_bal)

# Train Random Forest model
model_b = RandomForestClassifier(n_estimators=100, random_state=42)
model_b.fit(X_train_bal, y_train_bal)

# Save models and scaler
with open("models/model_a.pkl", "wb") as f:
    pickle.dump(model_a, f)

with open("models/model_b.pkl", "wb") as f:
    pickle.dump(model_b, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save encoders
for feature, le in encoders.items():
    with open(f"models/encoder_{feature}.pkl", "wb") as f:
        pickle.dump(le, f)

print("Models and scaler trained and saved!")
