from src.api import app

client = app.test_client()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json["status"] == "ok"
    assert "model_version" in r.json


def test_predict_setosa():
    r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    assert r.json["prediction"] == 0
    assert r.json["drift_alert"] is False


def test_drift_alert():
    r = client.post("/predict", json={"features": [15, 3.5, 12, 5]})
    assert r.json["drift_alert"] is True