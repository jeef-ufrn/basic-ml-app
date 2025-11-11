from fastapi.testclient import TestClient
import pytest

import app.app as myapp

client = TestClient(myapp.app)


class FakeModel:
    def predict(self, text: str):
        return "greet", {"greet": 0.9, "other": 0.1}


class FakeCollection:
    def __init__(self):
        self.last_insert = None

    def insert_one(self, doc):
        doc["_id"] = "fake-object-id"
        self.last_insert = doc
        return type("R", (), {"inserted_id": doc["_id"]})()


def setup_function():
    myapp.ENV = "dev"
    myapp.MODELS = {"testmodel": FakeModel()}
    myapp.collection = FakeCollection()


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()

    assert "message" in data
    assert isinstance(data["message"], str)


def test_predict_endpoint_inserts_and_returns_predictions():
    text = "hello world"
    resp = client.post("/predict", params={"text": text})
    assert resp.status_code == 200
    data = resp.json()

    assert data["text"] == text
    assert data["owner"] == "dev_user"
    assert "predictions" in data
    assert "testmodel" in data["predictions"]

    pm = data["predictions"]["testmodel"]
    assert pm["top_intent"] == "greet"
    assert isinstance(pm["all_probs"], dict)
    assert pm["all_probs"]["greet"] == pytest.approx(0.9)


    assert isinstance(myapp.collection.last_insert, dict)
    assert data["id"] == "fake-object-id"

    stored = myapp.collection.last_insert
    assert stored["text"] == text
    assert stored["owner"] == "dev_user"
    assert "predictions" in stored
    assert "timestamp" in stored
