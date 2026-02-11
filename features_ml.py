"""
Feature preparation, model training, evaluation, and prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


FEATURE_COLS = ["mean_ndvi", "distance_to_center", "area_km2"]
TARGET_COL = "mean_temp"


def prepare_features(df):
    """Extract feature matrix X and target vector y, dropping rows with NaN."""
    subset = df[FEATURE_COLS + [TARGET_COL]].dropna()
    X = subset[FEATURE_COLS]
    y = subset[TARGET_COL]
    n_dropped = len(df) - len(X)
    if n_dropped:
        print(f"  Dropped {n_dropped} tracts with missing data")
    print(f"  Using {len(X)} tracts with features: {', '.join(FEATURE_COLS)}")
    return X, y


def train_model(X_train, y_train):
    """Fit a Random Forest regressor to predict temperature from tract features."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return a dict with R², RMSE, and MAE on the test set."""
    y_pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
    }


def predict_all(model, df):
    """Generate temperature predictions for every row in df."""
    return pd.Series(
        model.predict(df[FEATURE_COLS]),
        index=df.index,
        name="predicted_temp",
    )
