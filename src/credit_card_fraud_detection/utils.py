import pandas as pd


def missing_value_summary(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()