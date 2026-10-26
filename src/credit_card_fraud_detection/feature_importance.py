import pandas as pd


def get_feature_importance(
    model,
    feature_names,
) -> pd.DataFrame:
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )

    return (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )
    
importance_df = get_feature_importance(
    rf_result.model,
    processed.X_train.columns,
)

print(
    importance_df.head(10)
)