import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")