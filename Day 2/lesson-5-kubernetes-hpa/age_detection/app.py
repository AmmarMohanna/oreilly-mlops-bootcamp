from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "model"
age_model = cv2.dnn.readNetFromCaffe(
    str(MODEL_DIR / "deploy_age.prototxt"),
    str(MODEL_DIR / "age_net.caffemodel"),
)
age_model_lock = Lock()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/stress")
def stress():
    """Run a synthetic inference so the HPA demo has meaningful CPU load."""
    image = np.zeros((227, 227, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False,
    )
    with age_model_lock:
        age_model.setInput(blob)
        age_model.forward()
    return jsonify({"status": "completed"})


@app.post("/detect_age")
def agedetection():
    try:
        image = request.files["image"]
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "The uploaded file is not a valid image"}), 400

        blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        with age_model_lock:
            age_model.setInput(blob)
            age_preds = age_model.forward()
        i = age_preds[0].argmax()
        age_interval = [
            "(0, 3)", "(4, 7)", "(8, 14)", "(15, 24)",
            "(25, 37)", "(38, 47)", "(48, 59)", "(60, 100)",
        ][i]

        return jsonify({"age_interval": age_interval})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
