from dataclasses import dataclass

import pandas as pd
from imblearn.over_sampling import SMOTE


@dataclass
class SmoteResult:
    X_train_resampled: pd.DataFrame
    y_train_resampled: pd.Series


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> SmoteResult:
    smote = SMOTE(
        random_state=42,
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train,
    )

    return SmoteResult(
        X_train_resampled=X_resampled,
        y_train_resampled=y_resampled,
    )