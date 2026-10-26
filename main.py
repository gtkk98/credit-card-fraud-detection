import pandas as pd

from src.credit_card_fraud_detection.preprocessing import preprocess_data
from src.credit_card_fraud_detection.train import train_logistic_regression
from src.credit_card_fraud_detection.evaluate import evaluate_model
from src.credit_card_fraud_detection.smote import apply_smote

df = pd.read_csv("data/creditcard.csv")

processed = preprocess_data(df)

training_result = train_logistic_regression(
    processed.X_train,
    processed.y_train,
)

evaluate_model(
    training_result.model,
    processed.X_test,
    processed.y_test,
)

# Apply SMOTE
smote_result = apply_smote(
    processed.X_train,
    processed.y_train,
)

print(
    smote_result.y_train_resampled.value_counts()
)

# Train with balanced data
training_result = train_logistic_regression(
    smote_result.X_train_resampled,
    smote_result.y_train_resampled,
)

# Temp
print(
    processed.y_train.value_counts()
)