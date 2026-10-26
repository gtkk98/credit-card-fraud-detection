import pandas as pd

from src.credit_card_fraud_detection.preprocessing import preprocess_data
from src.credit_card_fraud_detection.smote import apply_smote

def test_smote_balances_classes() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(1000),
            "V1": range(1000),
            "Class": [0] * 950 + [1] * 50,
        }
    )

    processed = preprocess_data(df)

    result = apply_smote(
        processed.X_train,
        processed.y_train,
    )

    counts = result.y_train_resampled.value_counts()

    assert counts[0] == counts[1]
    
def test_smote_increases_training_size() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(1000),
            "V1": range(1000),
            "Class": [0] * 950 + [1] * 50,
        }
    )

    processed = preprocess_data(df)

    result = apply_smote(
        processed.X_train,
        processed.y_train,
    )

    assert (
        len(result.X_train_resampled)
        > len(processed.X_train)
    )

def test_smote_returns_value() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 90 + [1] * 10,
        }
    )

    processed = preprocess_data(df)

    result = apply_smote(
        processed.X_train,
        processed.y_train,
    )

    print(result)

    assert result is not None