import pandas as pd

from src.credit_card_fraud_detection.preprocessing import preprocess_data
from src.credit_card_fraud_detection.smote import apply_smote
from src.credit_card_fraud_detection.tune_random_forest import tune_random_forest


def test_tuned_random_forest_returns_model() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 80 + [1] * 20,
        }
    )

    processed = preprocess_data(df)

    smote_result = apply_smote(
        processed.X_train,
        processed.y_train,
    )

    model = tune_random_forest(
        smote_result.X_train_resampled,
        smote_result.y_train_resampled,
    )

    assert model is not None
    
# Test Prediction Works

def test_tuned_random_forest_predicts_for_all_rows() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 80 + [1] * 20,
        }
    )

    processed = preprocess_data(df)

    smote_result = apply_smote(
        processed.X_train,
        processed.y_train,
    )

    model = tune_random_forest(
        smote_result.X_train_resampled,
        smote_result.y_train_resampled,
        n_iter=2,
    )

    predictions = model.predict(
        processed.X_test
    )

    assert len(predictions) == len(
        processed.y_test
    )

# Test Feature Importance Exists
def test_tuned_random_forest_has_feature_importance() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 80 + [1] * 20,
        }
    )

    processed = preprocess_data(df)

    smote_result = apply_smote(
        processed.X_train,
        processed.y_train,
    )

    model = tune_random_forest(
        smote_result.X_train_resampled,
        smote_result.y_train_resampled,
    )

    assert len(model.feature_importances_) == 2

# Test Feature Importances Sum to 1

def test_feature_importances_sum_to_one() -> None:
    df = pd.DataFrame(
        {
            "Amount": range(100),
            "V1": range(100),
            "Class": [0] * 80 + [1] * 20,
        }
    )

    processed = preprocess_data(df)

    smote_result = apply_smote(
        processed.X_train,
        processed.y_train,
    )

    model = tune_random_forest(
        smote_result.X_train_resampled,
        smote_result.y_train_resampled,
    )

    total_importance = (
        model.feature_importances_.sum()
    )

    assert abs(total_importance - 1.0) < 1e-10