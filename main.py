from pathlib import Path

import pandas as pd

from src.credit_card_fraud_detection.preprocessing import (
    preprocess_data,
)
from src.credit_card_fraud_detection.train import (
    train_logistic_regression,
)
from src.credit_card_fraud_detection.evaluate import (
    evaluate_model,
)
from src.credit_card_fraud_detection.smote import (
    apply_smote,
)
from src.credit_card_fraud_detection.train_random_forest import (
    train_random_forest,
)
from src.credit_card_fraud_detection.tune_random_forest import (
    tune_random_forest,
)
from src.credit_card_fraud_detection.model_io import (
    save_model,
    load_model,
)


DATA_PATH = Path("data/creditcard.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "random_forest.pkl"


def main() -> None:
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION")
    print("=" * 60)

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    print("\nPreprocessing data...")

    processed = preprocess_data(df)

    print("Preprocessing completed.")

    print(f"X_train shape: {processed.X_train.shape}")
    print(f"X_test shape : {processed.X_test.shape}")
    print(f"y_train shape: {processed.y_train.shape}")
    print(f"y_test shape : {processed.y_test.shape}")

    print("\n" + "=" * 60)
    print("BASELINE LOGISTIC REGRESSION")
    print("=" * 60)

    baseline_result = train_logistic_regression(
        processed.X_train,
        processed.y_train,
    )

    evaluate_model(
        baseline_result.model,
        processed.X_test,
        processed.y_test,
    )

    print("\n" + "=" * 60)
    print("APPLYING SMOTE")
    print("=" * 60)

    smote_result = apply_smote(
        processed.X_train,
        processed.y_train,
    )

    print("\nClass distribution before SMOTE:")
    print(processed.y_train.value_counts())

    print("\nClass distribution after SMOTE:")
    print(
        smote_result.y_train_resampled.value_counts()
    )

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION WITH SMOTE")
    print("=" * 60)

    smote_lr_result = train_logistic_regression(
        smote_result.X_train_resampled,
        smote_result.y_train_resampled,
    )

    evaluate_model(
        smote_lr_result.model,
        processed.X_test,
        processed.y_test,
    )

    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    print("Starting Random Forest")

    rf_result = train_random_forest(
        smote_result.X_train_resampled,
        smote_result.y_train_resampled,
    )

    print("Random Forest Completed")

    evaluate_model(
        rf_result.model,
        processed.X_test,
        processed.y_test,
    )

    print("\n" + "=" * 60)
    print("TUNING RANDOM FOREST")
    print("=" * 60)

    best_rf_model = tune_random_forest(
        smote_result.X_train_resampled,
        smote_result.y_train_resampled,
    )

    print("\nEvaluating tuned model...")

    evaluate_model(
        best_rf_model,
        processed.X_test,
        processed.y_test,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"\nSaving model to: {MODEL_PATH}")

    save_model(
        best_rf_model,
        MODEL_PATH,
    )

    print("Model saved successfully.")

    print("\nLoading saved model...")

    loaded_model = load_model(
        MODEL_PATH,
    )

    print(
        f"Loaded model type: {type(loaded_model).__name__}"
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()