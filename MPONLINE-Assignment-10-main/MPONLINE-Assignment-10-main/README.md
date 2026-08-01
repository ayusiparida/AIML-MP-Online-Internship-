# Assignment 10 — Heart Disease Prediction: End-to-End ML Deployment 

## Objective
Build a machine learning model that predicts whether a patient is at
risk of heart disease based on clinical parameters, expose it as a REST
API using Flask, and deploy it as a live, publicly accessible web
service on Render.

## Dataset
- Heart Disease Prediction Dataset (Kaggle):
  https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- Download `heart.csv` from the link above and place it in this folder
  before running `train_model.py`.
- If `heart.csv` is not found, the script automatically generates a
  small synthetic dataset with the same 13 clinical columns so the full
  pipeline (training, evaluation, serialization) can still be
  demonstrated end-to-end.

## Libraries Used
- `pandas`, `numpy` — data handling
- `scikit-learn` — `StandardScaler`, `LogisticRegression`, train/test
  split, accuracy scoring
- `joblib` — model/scaler serialization
- `flask` — REST API
- `gunicorn` — production WSGI server (used by Render)

## Methodology
1. **Data Understanding & Preprocessing** — Load `heart.csv`, display
   the first five records, identify the 13 numerical/clinical input
   features and the binary `target` variable, check for missing values,
   and split 80/20 into train/test sets.
2. **Model Development** — Standardize features with `StandardScaler`
   and train a **Logistic Regression** classifier. Evaluate with
   accuracy score, then serialize the model, scaler, and feature order
   with Joblib (`model.pkl`, `scaler.pkl`, `feature_columns.pkl`).
3. **API Development** — A Flask app (`app.py`) loads the saved model
   and scaler at startup and exposes:
   - `GET /` — health check / optional HTML form for manual testing
   - `POST /predict` — accepts patient details as JSON, returns a JSON
     prediction (`{"prediction": "Heart Disease Detected"}` or
     `{"prediction": "No Heart Disease Detected"}`), plus the predicted
     class probabilities.
4. **Deployment** — The app is deployed to Render as a web service using
   `gunicorn` (see `Procfile`) and `requirements.txt`.

## Model Performance
Logistic Regression test accuracy (printed by `train_model.py` when run
against the real Kaggle dataset — re-run locally to get your final
number; the number below is from the synthetic-fallback demonstration
run in development):

```
Test Accuracy: ~0.79 (synthetic fallback data)
```

## API Usage

**Request**
```bash
curl -X POST https://<your-render-app>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
      }'
```

**Response**
```json
{
  "prediction": "No Heart Disease Detected",
  "prediction_label": 0,
  "probability": { "no_disease": 0.64, "disease": 0.36 }
}
```

## Deployed Application

🔗 **Live Demo:** https://mponline-assignment-10.onrender.com

Visit the deployed application to:
- Access the web interface for manual heart disease prediction.
- Test the REST API endpoint (`/predict`) using JSON requests.
- Verify that the Flask application is running successfully on Render.

## Repository Structure
```
HeartDiseaseDeployment/
│
├── app.py                  # Flask REST API
├── train_model.py          # Data loading, preprocessing, training, serialization
├── model.pkl               # Trained Logistic Regression model
├── scaler.pkl              # Fitted StandardScaler
├── feature_columns.pkl     # Feature column order used at inference time
├── requirements.txt        # Python dependencies for Render
├── Procfile                # Render/gunicorn start command
├── README.md
├── heart.csv                # (Add locally; not committed — see Kaggle link above)
├── templates/
│   └── index.html          # Optional manual-testing web form
└── static/                 # (Optional, unused)
```

> **Note:** `heart.csv` itself is not committed to this repository —
> Kaggle's license does not clearly permit redistribution. Download it
> from the link above to retrain locally with the real data.

## How to Run Locally
```bash
pip install -r requirements.txt

# 1. Train the model (place heart.csv here first for real results)
python train_model.py

# 2. Run the API locally
python app.py
# API available at http://127.0.0.1:5000
```

## How to Deploy on Render
1. Push this repository to a **public GitHub repo**.
2. Go to [render.com](https://render.com) → **New +** → **Web Service**.
3. Connect your GitHub account and select this repository.
4. Configure the service:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (already set in `Procfile`)
5. Click **Create Web Service** and wait for the build/deploy to finish.
6. Once live, copy the Render URL (e.g.
   `https://heart-disease-predictor.onrender.com`) and:
   - Paste it into the **"Deployed Application"** section above.
   - Include it in your Google Form submission.
7. Test the live endpoint with the `curl` example above before
   submitting, to confirm it returns predictions successfully.

## Conclusion
This project delivered a complete MLOps pipeline: a Logistic Regression
model was trained on clinical features to predict heart disease risk,
serialized with Joblib, and wrapped in a Flask REST API that accepts
patient data as JSON and returns a prediction. The main challenges in
deployment were ensuring the exact feature order and scaling used during
training were reproduced identically at inference time, and configuring
a production-ready WSGI server (gunicorn) rather than Flask's built-in
development server for the live Render deployment. This project
highlights why MLOps matters in real-world machine learning: a model is
only useful once it can be reliably packaged, version-controlled, served
through an API, and kept running in production — bridging the gap
between a notebook experiment and a system that clinicians or downstream
applications can actually query.

## Author
**Sajjad Shaik**
GitHub: [SajjadShaik2005](https://github.com/SajjadShaik2005)
Email: sajjad102005@gmail.com
