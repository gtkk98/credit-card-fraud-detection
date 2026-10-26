import pandas as pd

from src.credit_card_fraud_detection.preprocessing import preprocess_data
from src.credit_card_fraud_detection.train_random_forest import train_random_forest


def test_random_forest_trains_successfully() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 80 + [1] * 20,
        }
    )

    processed = preprocess_data(df)

    result = train_random_forest(
        processed.X_train,
        processed.y_train,
    )

    assert result.model is not None
    
def test_random_forest_predicts_for_all_rows() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 80 + [1] * 20,
        }
    )

    processed = preprocess_data(df)

    result = train_random_forest(
        processed.X_train,
        processed.y_train,
    )

    predictions = result.model.predict(
        processed.X_test
    )

    assert len(predictions) == len(
        processed.y_test
    )