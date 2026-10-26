from pathlib import Path

import joblib


def save_model(
    model,
    path: Path,
) -> None:
    joblib.dump(
        model,
        path,
    )


def load_model(
    path: Path,
):
    return joblib.load(path)