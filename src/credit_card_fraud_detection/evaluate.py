from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model,
    X_test,
    y_test,
):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
    )

    recall = recall_score(
        y_test,
        y_pred,
    )

    f1 = f1_score(
        y_test,
        y_pred,
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
