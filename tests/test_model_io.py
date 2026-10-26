from pathlib import Path

import pandas as pd

from src.credit_card_fraud_detection.model_io import (
    load_model,
    save_model,
)
from src.credit_card_fraud_detection.preprocessing import preprocess_data
from src.credit_card_fraud_detection.train import train_logistic_regression


def test_model_can_be_saved_and_loaded(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 80 + [1] * 20,
        }
    )

    processed = preprocess_data(df)

    training_result = train_logistic_regression(
        processed.X_train,
        processed.y_train,
    )

    model_path = (
        tmp_path / "model.pkl"
    )

    save_model(
        training_result.model,
        model_path,
    )

    loaded_model = load_model(
        model_path,
    )

    assert loaded_model is not None