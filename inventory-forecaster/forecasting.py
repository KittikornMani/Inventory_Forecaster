"""Inference for the single-series M5 LSTM/GRU Kaggle experiment."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

FEATURES = [
    "sales", "sell_price", "wday", "month", "year", "is_event", "snap",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_28",
]
LEGACY_FEATURES = ["sales", "sell_price", "wday", "month", "year", "is_event", "snap"]
REQUIRED = {"date", "sales", "sell_price", "is_event", "snap", "series_id"}
WINDOW = 56


def validate_upload(frame: pd.DataFrame, expected_series: str) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    data = frame.copy()
    if data.empty:
        raise ValueError("The CSV is empty.")
    ids = data["series_id"].dropna().astype(str).unique()
    if len(ids) != 1 or ids[0] != expected_series:
        raise ValueError(
            f"This model was trained for {expected_series}. Upload only that series."
        )
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ValueError("Every date must be a valid date, such as 2020-01-31.")
    if data["date"].duplicated().any():
        raise ValueError("Use exactly one row per date.")
    data = data.sort_values("date").reset_index(drop=True)
    if not data["date"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("Sales history must contain consecutive calendar days.")
    for column in ("sales", "sell_price", "is_event", "snap"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"{column} must contain numbers on every day.")
    if (data[["sales", "sell_price"]] < 0).any().any():
        raise ValueError("Sales and sell_price cannot be negative.")
    if not data[["is_event", "snap"]].isin([0, 1]).all().all():
        raise ValueError("is_event and snap must contain only 0 or 1.")
    if len(data) < 84:
        raise ValueError("Upload at least 84 consecutive days (28 feature days + 56 input days).")
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    # M5 wday is Saturday=1 through Friday=7.
    out["wday"] = (out["date"].dt.dayofweek + 2) % 7 + 1
    out["month"] = out["date"].dt.month
    out["year"] = out["date"].dt.year
    for lag in (1, 7, 14, 28):
        out[f"lag_{lag}"] = out["sales"].shift(lag)
    out["rolling_mean_7"] = out["sales"].rolling(7).mean()
    out["rolling_mean_28"] = out["sales"].rolling(28).mean()
    out["rolling_std_28"] = out["sales"].rolling(28).std()
    return out.dropna(subset=FEATURES).reset_index(drop=True)


def load_artifacts(folder: Path):
    files = {
        "LSTM": folder / "lstm_inventory_forecaster.keras",
        "GRU": folder / "gru_inventory_forecaster.keras",
        "features": folder / "feature_scaler.pkl",
        "target": folder / "target_scaler.pkl",
    }
    missing = [str(path.name) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Add these files to models/: " + ", ".join(missing))
    models = {name: load_model(files[name], compile=False) for name in ("LSTM", "GRU")}
    feature_scaler = joblib.load(files["features"])
    target_scaler = joblib.load(files["target"])
    dimensions = {name: tuple(model.input_shape[1:]) for name, model in models.items()}
    if len(set(dimensions.values())) != 1:
        raise ValueError("The LSTM and GRU models have different input shapes.")
    shape = next(iter(dimensions.values()))
    if shape == (56, 14):
        window_size, feature_columns, log_target = 56, FEATURES, True
    elif shape == (28, 7):
        window_size, feature_columns, log_target = 28, LEGACY_FEATURES, False
    else:
        raise ValueError(f"Unsupported model input shape {shape}. Expected (28, 7) or (56, 14).")
    if getattr(feature_scaler, "n_features_in_", None) != len(feature_columns):
        raise ValueError("Feature scaler does not match the model input features.")
    if getattr(target_scaler, "n_features_in_", None) != 1:
        raise ValueError("Target scaler must have one output feature.")
    return models, feature_scaler, target_scaler, window_size, feature_columns, log_target


def forecast(
    raw: pd.DataFrame,
    model,
    feature_scaler,
    target_scaler,
    horizon: int,
    future_calendar: pd.DataFrame | None = None,
    window_size: int = 56,
    feature_columns: list[str] | None = None,
    log_target: bool = True,
) -> pd.DataFrame:
    """Recursive forecast; the new row is recomputed from all known/predicted sales."""
    history = raw.copy().sort_values("date").reset_index(drop=True)
    feature_data = add_features(history)
    feature_columns = feature_columns or FEATURES
    window = feature_scaler.transform(feature_data[feature_columns].iloc[-window_size:].to_numpy())
    rows = []
    calendar_by_date = None
    if future_calendar is not None:
        calendar_by_date = future_calendar.set_index("date")

    for day in range(1, horizon + 1):
        scaled = float(model.predict(window[np.newaxis, :, :], verbose=0)[0, 0])
        restored = float(target_scaler.inverse_transform([[scaled]])[0, 0])
        units = max(0.0, float(np.expm1(restored) if log_target else restored))
        next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
        event, snap = 0, 0
        if calendar_by_date is not None and next_date in calendar_by_date.index:
            event = int(calendar_by_date.loc[next_date, "is_event"])
            snap = int(calendar_by_date.loc[next_date, "snap"])
        new_row = pd.DataFrame([{
            "series_id": history["series_id"].iloc[0],
            "date": next_date,
            "sales": units,
            "sell_price": float(history["sell_price"].iloc[-1]),
            "is_event": event,
            "snap": snap,
        }])
        history = pd.concat([history, new_row], ignore_index=True)
        rows.append({"date": next_date, "forecast_day": day, "predicted_sales": units})
        features = add_features(history)
        next_scaled = feature_scaler.transform(features[feature_columns].iloc[[-1]].to_numpy())
        window = np.vstack([window[1:], next_scaled])

    return pd.DataFrame(rows)
