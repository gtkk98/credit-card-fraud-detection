from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier


@dataclass
class RandomForestTrainingResult:
    model: RandomForestClassifier


def train_random_forest(
    X_train,
    y_train,
) -> RandomForestTrainingResult:
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return RandomForestTrainingResult(
        model=model,
    )