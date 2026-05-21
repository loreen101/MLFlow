"""
Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""

from litestar.testing import TestClient

from app.model_utils import predict_churn
from main import app


# ---------------------------------------------------------------------------
# Function Tests
# ---------------------------------------------------------------------------


def test_predict_churn_returns_binary_value() -> None:
    sample_features = [600, "France", "Female", 40, 3, 60000, 2, 1, 1, 50000]
    prediction = predict_churn(sample_features)
    assert prediction in {0, 1}


def test_predict_churn_handles_numeric_edge_case() -> None:
    sample_features = [300, "Spain", "Male", 18, 0, 0, 1, False, False, 0]
    prediction = predict_churn(sample_features)
    assert prediction in {0, 1}


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------


def test_get_root_returns_welcome_message() -> None:
    with TestClient(app=app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the churn prediction API"}


def test_get_health_returns_healthy_status() -> None:
    with TestClient(app=app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_post_predict_returns_prediction() -> None:
    payload = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000,
    }
    with TestClient(app=app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in {0, 1}


def test_post_predict_with_invalid_data_returns_400() -> None:
    payload = {
        "CreditScore": "invalid",
        "Geography": "France",
        "Gender": "Female",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": True,
        "IsActiveMember": True,
        "EstimatedSalary": 50000,
    }
    with TestClient(app=app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 400
