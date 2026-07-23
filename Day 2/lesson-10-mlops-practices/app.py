from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load("model/rf_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

# Prometheus Metrics
REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests")
REQUEST_EXCEPTIONS = Counter("predict_exceptions_total", "Total exceptions in predictions")
REQUEST_LATENCY = Histogram("predict_request_latency_seconds", "Latency for prediction requests")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
@REQUEST_LATENCY.time()
def predict():
    REQUEST_COUNT.inc()
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided or invalid format"}), 400

        data = pd.DataFrame([data])
        processed_data = preprocessor.transform(data)
        prediction = model.predict(processed_data)

        label_map = {0: "<=50K", 1: ">50K"}
        predicted_label = label_map[int(prediction[0])]

        return jsonify({"prediction": predicted_label})
    except Exception as e:
        REQUEST_EXCEPTIONS.inc()
        return jsonify({"error": str(e)}), 500

# Metrics endpoint for Prometheus
@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
