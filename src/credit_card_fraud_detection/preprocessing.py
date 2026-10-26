from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
@dataclass
class PreprocessingResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    
def validate_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found.")
    
def preprocess_data(df: pd.DataFrame,) -> PreprocessingResult:
    validate_dataframe(df)
    
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X.columns,
    )

    return PreprocessingResult(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
    )
   