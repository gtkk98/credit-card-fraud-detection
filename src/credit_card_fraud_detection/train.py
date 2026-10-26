from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

@dataclass
class TrainingResult:
    model: LogisticRegression
    
def train_logistic_regression(
    X_train,
    y_train,
) -> TrainingResult:
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )
    
    model.fit(
        X_train,
        y_train,
    )
    
    return TrainingResult(
        model=model,
    )