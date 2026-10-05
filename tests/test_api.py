"""Tests for API endpoints."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_predict_valid_input():
    payload = {
        "age": 30, "income": 50000, "loan_amount": 15000,
        "credit_score": 720, "employment_years": 5,
        "num_previous_loans": 1, "has_mortgage": 0, "num_dependents": 0,
    }
    r = client.post("/predict", json=payload)
    if r.status_code == 503:
        return  # model not trained
    assert r.status_code == 200
    body = r.json()
    assert "default_probability" in body
    assert 0 <= body["default_probability"] <= 1
    assert body["risk_level"] in ("Low", "Medium", "High")


def test_predict_invalid_age():
    payload = {"age": 5, "income": 50000, "loan_amount": 15000,
               "credit_score": 720, "employment_years": 5}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422  # validation error
