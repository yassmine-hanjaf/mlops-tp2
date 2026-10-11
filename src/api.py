from flask import Flask, request, jsonify
import joblib
from sklearn.datasets import load_iris

MODEL_VERSION = "1.0"

app = Flask(__name__)
model = joblib.load("models/iris_model.pkl")

# Statistiques d'entraînement, pour l'alerte de drift
iris = load_iris()
TRAIN_MEAN = iris.data.mean(axis=0)
TRAIN_STD = iris.data.std(axis=0)
FEATURES = iris.feature_names


@app.get("/")
def home():
    return jsonify({"service": "Iris MLOps API", "status": "running"})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_version": MODEL_VERSION})


@app.post("/predict")
def predict():
    d = request.get_json()
    features = d["features"]
    p = model.predict([features])

    # Alerte simple : une valeur à plus de 3 écarts-types de la moyenne d'entraînement
    drifted = [
        FEATURES[i]
        for i, v in enumerate(features)
        if abs(v - TRAIN_MEAN[i]) > 3 * TRAIN_STD[i]
    ]
    return jsonify({
        "prediction": int(p[0]),
        "model_version": MODEL_VERSION,
        "drift_alert": len(drifted) > 0,
        "drifted_features": drifted,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)