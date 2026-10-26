from fastapi import FastAPI

from api.predictor import (
    predict_transaction,
)
from api.schemas import (
    PredictionResponse,
    TransactionRequest,
)

app = FastAPI(
    title="Credit Card Fraud Detection API",
)

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
    
@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: TransactionRequest,
):
    prediction, probability = (
        predict_transaction(
            request.model_dump()
        )
    )

    return PredictionResponse(
        prediction=prediction,
        fraud_probability=probability,
    )