"""
AI-ML Assignment - 10
Flask REST API for Heart Disease Prediction

Loads the trained model + scaler (produced by train_model.py) and
exposes a /predict endpoint that accepts patient clinical data as JSON
and returns a prediction as JSON.
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "feature_columns.pkl"

model = None
scaler = None
feature_columns = None


def load_artifacts():
    """Load the trained model, scaler, and feature order once at startup."""
    global model, scaler, feature_columns
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        raise FileNotFoundError(
            "model.pkl / scaler.pkl not found. Run 'python train_model.py' first."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)


load_artifacts()


@app.route("/", methods=["GET"])
def home():
    """Simple health-check / optional web form for manual testing."""
    if os.path.exists(os.path.join("templates", "index.html")):
        return render_template("index.html", features=feature_columns)
    return jsonify({
        "status": "ok",
        "message": "Heart Disease Prediction API is running.",
        "usage": "POST patient data as JSON to /predict",
        "expected_fields": feature_columns,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts patient details as JSON, e.g.:
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    Returns:
    { "prediction": "Heart Disease Detected" }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    if data is None:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    missing = [f for f in feature_columns if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        input_values = [float(data[f]) for f in feature_columns]
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid field value: {e}"}), 400

    input_df = pd.DataFrame([input_values], columns=feature_columns)
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0].tolist()

    result = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"

    return jsonify({
        "prediction": result,
        "prediction_label": int(pred),
        "probability": {"no_disease": proba[0], "disease": proba[1]},
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
