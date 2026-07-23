import requests
import pandas as pd

url = "http://localhost:8000/predict"

results = []
data = pd.read_csv("diabetes_prediction_dataset.csv")
samples = data.sample(n=100, random_state=42)

for _, row in samples.iterrows():
    sample = {
        "features": {
            "age": float(row["age"]),
            "bmi": float(row["bmi"]),
            "HbA1c_level": float(row["HbA1c_level"]),
            "blood_glucose_level": int(row["blood_glucose_level"]),
            "gender": str(row["gender"]),
            "smoking_history": str(row["smoking_history"]),
            "hypertension": int(row["hypertension"]),
            "heart_disease": int(row["heart_disease"]),
        },
        "true_label": int(row["diabetes"]),
    }

    response = requests.post(url, json=sample, timeout=10)
    response.raise_for_status()
    results.append(response.json())

print(f"Sent {len(results)} requests successfully.")
