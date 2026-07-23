from fastapi import FastAPI, HTTPException, Request
import pickle
import pandas as pd
import uvicorn

# Load models and scaler
with open("models/model_a.pkl", "rb") as f:
    model_a = pickle.load(f)

with open("models/model_b.pkl", "rb") as f:
    model_b = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load dataset to identify columns
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'smoking_history']
other_features = ['hypertension', 'heart_disease']

# Load encoders for categorical features
encoders = {}
for feature in categorical_features:
    with open(f"models/encoder_{feature}.pkl", "rb") as f:
        encoders[feature] = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Store logs in memory
logs = []

# Counter to alternate models A/B
counter = 0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    global counter
    counter += 1
    # Alternating between model A and B
    assigned_model = "A" if counter % 2 == 0 else "B"

    data = await request.json()
    feature_dict = data.get("features") 
    if not feature_dict:
        raise HTTPException(status_code=400, detail="'features' is required")

    numerical_data = pd.DataFrame(
        [{col: float(feature_dict[col]) for col in numerical_features}],
        columns=numerical_features,
    )
    numerical_data_scaled = scaler.transform(numerical_data)[0]

    # Prepare categorical data (apply label encoding)
    categorical_data = []
    for col in categorical_features:
        le = encoders[col]
        try:
            encoded_val = le.transform([feature_dict[col]])[0]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid category '{feature_dict[col]}' for feature '{col}'. "
                    f"Expected one of {list(le.classes_)}"
                ),
            )
        categorical_data.append(encoded_val)

    processed_values = dict(zip(numerical_features, numerical_data_scaled))
    processed_values.update(dict(zip(categorical_features, categorical_data)))
    processed_values.update(
        {col: float(feature_dict[col]) for col in other_features}
    )
    model_feature_order = list(model_a.feature_names_in_)
    if model_feature_order != list(model_b.feature_names_in_):
        raise RuntimeError("Model A and model B use different feature orders")
    processed_features = pd.DataFrame(
        [processed_values], columns=model_feature_order
    )

    # Get prediction from the selected model
    if assigned_model == "A":
        prediction = int(model_a.predict(processed_features)[0])
    else:
        prediction = int(model_b.predict(processed_features)[0])

    # Log result
    log_entry = {
        "input": feature_dict,
        "model": assigned_model,
        "prediction": prediction,
        "true_label": data.get("true_label")
    }
    logs.append(log_entry)

    return {"model": assigned_model, "prediction": prediction}

@app.get("/logs")
def get_logs():
    return logs

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
