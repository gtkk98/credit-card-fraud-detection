from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def tune_random_forest(
    X_train,
    y_train,
    n_iter: int = 10,
):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
    )

    param_distributions = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=3,
    )

    search.fit(
        X_train,
        y_train,
    )

    print("\nBest Parameters")
    print(search.best_params_)

    return search.best_estimator_