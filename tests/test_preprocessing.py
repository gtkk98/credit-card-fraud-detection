import pandas as pd
import pytest

from src.credit_card_fraud_detection.preprocessing import validate_dataframe
from src.credit_card_fraud_detection.preprocessing import preprocess_data

"""y = df["Class"] would fail later.

We want validation to catch the problem early."""

def test_validate_dataframe_raises_error_for_empty_dataframe() -> None:
    df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        validate_dataframe(df)

# Missing Class Columns
        
def test_validate_dataframe_raises_error_when_class_column_missing() -> None:
    df = pd.DataFrame(
        {
            "Amount": [100, 200],
            "V1": [0.5, 0.8]
        }
    )
    
    with pytest.raises(ValueError):
        validate_dataframe(df)

# Valid DataFrame

def test_validate_dataframe_accepts_valid_dataframe() -> None:
    df = pd.DataFrame(
        {
            "Amount": [100, 200],
            "V1": [0.5, 0.8],
            "Class": [0, 1],
        }
    )
    
    validate_dataframe(df)

# Train/Test Split Sizes

def test_preprocess_data_creates_train_test_split() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 90 + [1] * 10,
        }
    )
    
    result = preprocess_data(df)
    
    assert len(result.X_train) == 80
    assert len(result.X_test) == 20

# Target Column Removed

def test_preprocess_data_removes_target_column() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 90 + [1] * 10,
        }
    )
    
    result = preprocess_data(df)
    
    assert "Class" not in result.X_train.columns
    assert "Class" not in result.X_test.columns
    
# Check Column Count

def test_preprocess_data_keeps_expected_feature_count() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 90 + [1] * 10,
        }
    )

    result = preprocess_data(df)

    assert result.X_train.shape[1] == 2

# Scaling Actually Happens

def test_scaled_features_have_mean_close_to_zero() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100, 200),
            "Class": [0] * 90 + [1] * 10,
        }
    )

    result = preprocess_data(df)

    amount_mean = result.X_train["Amount"].mean()

    assert abs(amount_mean) < 1e-10

# Class Ratio Preserved

def test_stratify_preserves_class_distribution() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(1000),
            "V1": range(1000),
            "Class": [0] * 950 + [1] * 50,
        }
    )

    result = preprocess_data(df)

    train_ratio = result.y_train.mean()
    test_ratio = result.y_test.mean()

    assert abs(train_ratio - test_ratio) < 0.01