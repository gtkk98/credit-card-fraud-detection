from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "models/random_forest.pkl"
)

model = joblib.load(
    MODEL_PATH
)

def predict_transaction(
    transaction: dict,
):
    df = pd.DataFrame(
        [transaction]
    )

    prediction = model.predict(
        df
    )[0]

    probability = (
        model.predict_proba(df)[0][1]
    )

    return (
        int(prediction),
        float(probability),
    )
