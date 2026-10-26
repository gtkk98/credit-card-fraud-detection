import pandas as pd

from src.credit_card_fraud_detection.preprocessing import preprocess_data
from src.credit_card_fraud_detection.train import train_logistic_regression

def test_logistic_regression_trains_successfully() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 80 + [1] * 20,
        }
    )
    
    processed = preprocess_data(df)
    
    result = train_logistic_regression(
        processed.X_train,
        processed.y_train,
    )
    
    assert result.model is not None

# Test Prediction Count

def train_prediction_count_matches_test_size() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 80 + [1] * 20,
        }
    )
    
    processed = preprocess_data(df)
    
    result = train_logistic_regression(
        processed.X_train,
        processed.y_train,
    )
    
    predictions = result.model.predict(
        processed.X_test
    )
    
    assert len(predictions) == len(processed.y_test)