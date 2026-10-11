from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("models/iris_model.pkl")

@app.get("/")
def home():
    return jsonify({"service": "Iris MLOps API", "status": "running"})

@app.post("/predict")
def predict():
    d = request.get_json()
    p = model.predict([d["features"]])
    return jsonify({"prediction": int(p[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)