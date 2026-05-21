"""
Churn Prediction API

Run with:
    litestar --app main:app run --reload
Then open:
    http://localhost:8000/schema/swagger
"""

from litestar import Litestar, get, post, Response
from pydantic import BaseModel
from app.model_utils import predict_churn
from app.logger_setup import setup_logging

logger = setup_logging()


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
class ChurnRequest(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@get("/", sync_to_thread=False)
def home() -> dict[str, str]:
    """GET / - Welcome message for the churn prediction API."""
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the churn prediction API"}


@get("/health", sync_to_thread=False)
def health() -> dict[str, str]:
    """GET /health - Health check endpoint."""
    logger.info("Health endpoint accessed")
    return {"status": "healthy"}


@post("/predict", sync_to_thread=False)
def predict(data: ChurnRequest) -> dict[str, int]:
    try:
        features = [
            data.CreditScore,
            data.Geography,
            data.Gender,
            data.Age,
            data.Tenure,
            data.Balance,
            data.NumOfProducts,
            data.HasCrCard,
            data.IsActiveMember,
            data.EstimatedSalary,
        ]
        prediction = predict_churn(features)
        logger.info(f"Prediction made for features: {features} - Result: {prediction}")
        return Response({"prediction": int(prediction)}, status_code=200)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return Response({"error": "Invalid input"}, status_code=400)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Litestar(route_handlers=[home, health, predict])
