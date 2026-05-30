"""
Pearls AQI Predictor — Inference Engine
========================================
Provides helper functions consumed by the Streamlit dashboard:
  • load_model_from_registry()   – download model artefacts from Hopsworks
  • get_recent_features_from_hopsworks() – last 24 h of feature data
  • get_weather_forecast()       – hourly weather + AQ forecast from Open-Meteo
  • recursive_forecast()         – 72-hour recursive AQI prediction
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import joblib

# ── Make project root importable ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    AIR_QUALITY_API_URL,
    AQ_PARAMS,
    DERIVED_FEATURES,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FORECAST_HOURS,
    FORECAST_WEATHER_API_URL,
    LAG_HOURS,
    LATITUDE,
    LONGITUDE,
    MODEL_NAME,
    MODEL_VERSION,
    ROLLING_WINDOWS,
    TARGET_COLUMN,
    TIME_FEATURES,
    TIMEZONE,
    WEATHER_PARAMS,
    get_aqi_category,
)

warnings.filterwarnings("ignore")


# ======================================================================
# 1. Load model from Hopsworks Model Registry
# ======================================================================

def load_model_from_registry() -> Tuple[Any, Dict, Dict, List[str]]:
    """
    Download the model artefacts from Hopsworks and return
    (model, metrics, feature_importance, feature_columns).
    """
    import hopsworks

    print("🔗  Connecting to Hopsworks for model download …")
    project = hopsworks.login()
    mr = project.get_model_registry()

    print(f"📥  Fetching latest version of model '{MODEL_NAME}' …")
    all_models = mr.get_models(MODEL_NAME)
    hw_model = max(all_models, key=lambda m: m.version)
    model_dir = hw_model.download()

    model = joblib.load(os.path.join(model_dir, "model.pkl"))

    with open(os.path.join(model_dir, "metrics.json")) as f:
        metrics: Dict = json.load(f)

    with open(os.path.join(model_dir, "feature_importance.json")) as f:
        feature_importance: Dict = json.load(f)

    with open(os.path.join(model_dir, "feature_columns.json")) as f:
        feature_columns: List[str] = json.load(f)

    print(f"✅  Model loaded ({len(feature_columns)} features)")
    return model, metrics, feature_importance, feature_columns


# ======================================================================
# 2. Recent features from Hopsworks
# ======================================================================

def get_recent_features_from_hopsworks() -> pd.DataFrame:
    """
    Return the last 24 hours of feature data from the Hopsworks
    feature group, sorted ascending by timestamp.
    """
    import hopsworks

    print("🔗  Connecting to Hopsworks for recent features …")
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        cutoff = df["timestamp"].max() - pd.Timedelta(hours=24)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    print(f"✅  Recent features: {len(df)} rows")
    return df


# ======================================================================
# 3. Weather + AQ forecast from Open-Meteo
# ======================================================================

def _fetch_json(url: str, params: Dict) -> Optional[Dict]:
    """Send GET request and return parsed JSON, or None on failure."""
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"⚠️  API request failed ({url}): {exc}")
        return None


def get_weather_forecast(forecast_days: int = 3) -> pd.DataFrame:
    """
    Fetch hourly weather + air-quality forecasts from Open-Meteo for
    the next *forecast_days* days, merge into a single DataFrame with a
    'timestamp' column (UTC).
    """

    # ── Weather forecast ──────────────────────────────────────────────
    weather_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_PARAMS),
        "timezone": "UTC",
        "forecast_days": forecast_days,
    }
    weather_json = _fetch_json(FORECAST_WEATHER_API_URL, weather_params)

    # ── Air-quality forecast ──────────────────────────────────────────
    aq_forecast_params_list = [p for p in AQ_PARAMS if p != TARGET_COLUMN]
    # Also request us_aqi for reference
    aq_all = AQ_PARAMS.copy()
    aq_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(aq_all),
        "timezone": "UTC",
        "forecast_days": forecast_days,
        "past_days": 0,
    }
    aq_json = _fetch_json(AIR_QUALITY_API_URL, aq_params)

    # ── Parse weather hourly data ─────────────────────────────────────
    frames: List[pd.DataFrame] = []

    if weather_json and "hourly" in weather_json:
        hourly = weather_json["hourly"]
        wdf = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"], utc=True)})
        for param in WEATHER_PARAMS:
            if param in hourly:
                wdf[param] = hourly[param]
        frames.append(wdf)

    # ── Parse AQ hourly data ──────────────────────────────────────────
    if aq_json and "hourly" in aq_json:
        hourly = aq_json["hourly"]
        aqdf = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"], utc=True)})
        for param in aq_all:
            if param in hourly:
                aqdf[param] = hourly[param]
        frames.append(aqdf)

    if not frames:
        print("⚠️  No forecast data returned — returning empty DataFrame.")
        return pd.DataFrame()

    # Merge on timestamp
    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.merge(extra, on="timestamp", how="outer", suffixes=("", "_aq"))
    # Drop duplicate columns that end with _aq (keep original)
    dup_cols = [c for c in merged.columns if c.endswith("_aq")]
    merged = merged.drop(columns=dup_cols)

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged = merged.set_index("timestamp")

    print(f"✅  Weather+AQ forecast: {len(merged)} hourly rows, "
          f"{len(merged.columns)} features")
    return merged


# ======================================================================
# 4. Recursive 72-hour AQI forecast
# ======================================================================

def _compute_time_features(ts: pd.Timestamp) -> Dict[str, float]:
    """Derive time features from a single timestamp."""
    return {
        "hour": float(ts.hour),
        "day_of_week": float(ts.dayofweek),
        "month": float(ts.month),
        "day_of_month": float(ts.day),
        "is_weekend": float(ts.dayofweek >= 5),
        "hour_sin": float(np.sin(2 * np.pi * ts.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * ts.hour / 24)),
        "day_of_week_sin": float(np.sin(2 * np.pi * ts.dayofweek / 7)),
        "day_of_week_cos": float(np.cos(2 * np.pi * ts.dayofweek / 7)),
        "month_sin": float(np.sin(2 * np.pi * ts.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * ts.month / 12)),
    }


def _compute_derived_features(
    aqi_series: List[float],
    row_vals: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute derived features from the running AQI series and the
    current row values (weather params, etc.).
    """
    series = np.array(aqi_series, dtype=float)

    # AQI change rate (difference from previous)
    if len(series) >= 2:
        aqi_change_rate = float(series[-1] - series[-2])
    else:
        aqi_change_rate = 0.0

    # Rolling stats
    def _rolling(window: int) -> Tuple[float, float]:
        chunk = series[-window:] if len(series) >= window else series
        return float(np.mean(chunk)), float(np.std(chunk)) if len(chunk) > 1 else 0.0

    mean_6, std_6 = _rolling(6)
    mean_24, std_24 = _rolling(24)

    # Wind chill factor (simplified)
    temp = row_vals.get("temperature_2m", 25.0)
    wind = row_vals.get("windspeed_10m", 0.0)
    wind_chill = temp - (0.4 * wind) if wind > 0 else temp

    # Pollution index — mean of available sub-AQI values
    sub_aqi_keys = [
        "us_aqi_pm2_5", "us_aqi_pm10",
        "us_aqi_nitrogen_dioxide", "us_aqi_carbon_monoxide",
        "us_aqi_ozone", "us_aqi_sulphur_dioxide",
    ]
    sub_vals = [row_vals[k] for k in sub_aqi_keys if k in row_vals and not np.isnan(row_vals.get(k, np.nan))]
    pollution_index = float(np.mean(sub_vals)) if sub_vals else float(series[-1])

    return {
        "aqi_change_rate": aqi_change_rate,
        "aqi_rolling_mean_6h": mean_6,
        "aqi_rolling_mean_24h": mean_24,
        "aqi_rolling_std_6h": std_6,
        "aqi_rolling_std_24h": std_24,
        "wind_chill_factor": wind_chill,
        "pollution_index": pollution_index,
    }


def recursive_forecast(
    model: Any,
    feature_columns: List[str],
    recent_data: pd.DataFrame,
    weather_forecast: pd.DataFrame,
    n_hours: int = FORECAST_HOURS,
) -> pd.DataFrame:
    """
    Produce an *n_hours*-ahead AQI forecast using recursive
    single-step predictions.

    Parameters
    ----------
    model : trained sklearn/xgb estimator
    feature_columns : list of column names the model expects
    recent_data : last 24 h of Hopsworks feature data (must contain
        'timestamp' and 'us_aqi' columns as well as raw features)
    weather_forecast : hourly DataFrame returned by
        get_weather_forecast(), indexed by UTC timestamp
    n_hours : prediction horizon (default 72)

    Returns
    -------
    DataFrame with columns: timestamp, predicted_aqi, aqi_category,
        aqi_color
    """
    # --- Prepare recent AQI series (up to 24 recent values) -----------
    recent = recent_data.copy()
    if "timestamp" in recent.columns:
        recent["timestamp"] = pd.to_datetime(recent["timestamp"], utc=True)
        recent = recent.sort_values("timestamp").reset_index(drop=True)

    aqi_series: List[float] = recent[TARGET_COLUMN].dropna().tolist()
    if not aqi_series:
        aqi_series = [0.0]  # fallback

    # Determine the start timestamp
    if "timestamp" in recent.columns and len(recent) > 0:
        last_ts = recent["timestamp"].iloc[-1]
    else:
        last_ts = pd.Timestamp.now(tz="UTC").floor("h")

    # Cache the last row of recent data for fallback values
    last_row_dict: Dict[str, float] = {}
    if len(recent) > 0:
        for col in recent.columns:
            if col not in ("city", "timestamp") and pd.api.types.is_numeric_dtype(recent[col]):
                val = recent[col].iloc[-1]
                if pd.notna(val):
                    last_row_dict[col] = float(val)

    # --- Recursive loop -----------------------------------------------
    results: List[Dict] = []

    for step in range(1, n_hours + 1):
        current_ts = last_ts + pd.Timedelta(hours=step)

        # A) Weather + AQ forecast values for this hour
        row_vals: Dict[str, float] = dict(last_row_dict)  # start with last-known

        if not weather_forecast.empty:
            # Find the closest forecast hour
            if current_ts in weather_forecast.index:
                fc_row = weather_forecast.loc[current_ts]
            else:
                # Nearest-neighbour lookup
                idx = weather_forecast.index.get_indexer([current_ts], method="nearest")
                if idx[0] >= 0:
                    fc_row = weather_forecast.iloc[idx[0]]
                else:
                    fc_row = pd.Series(dtype=float)

            for col in fc_row.index:
                val = fc_row[col]
                if pd.notna(val):
                    row_vals[col] = float(val)

        # B) Time features
        time_feats = _compute_time_features(current_ts)
        row_vals.update(time_feats)

        # C) Lag features — use actual recent values or prior predictions
        for lag in LAG_HOURS:
            key = f"us_aqi_lag_{lag}h"
            idx = len(aqi_series) - lag
            if idx >= 0:
                row_vals[key] = aqi_series[idx]
            else:
                # Not enough history — use earliest available
                row_vals[key] = aqi_series[0]

        # D) Derived / rolling features from the running AQI series
        derived = _compute_derived_features(aqi_series, row_vals)
        row_vals.update(derived)

        # E) Build feature vector in the correct column order
        feature_vec: List[float] = []
        for col in feature_columns:
            val = row_vals.get(col, np.nan)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                # Fill missing with 0 (safe neutral value)
                val = 0.0
            feature_vec.append(float(val))

        # F) Predict
        try:
            X_step = pd.DataFrame([feature_vec], columns=feature_columns)
            pred_aqi = float(model.predict(X_step)[0])
        except Exception as exc:
            print(f"⚠️  Prediction failed at step {step}: {exc}")
            pred_aqi = float(aqi_series[-1])  # fallback to last known

        # Clamp to [0, 500]
        pred_aqi = float(np.clip(pred_aqi, 0, 500))

        # G) Update running series
        aqi_series.append(pred_aqi)

        # H) Build output row
        category, color = get_aqi_category(pred_aqi)
        results.append({
            "timestamp": current_ts,
            "predicted_aqi": round(pred_aqi, 2),
            "aqi_category": category,
            "aqi_color": color,
        })

    forecast_df = pd.DataFrame(results)
    print(f"✅  Recursive forecast complete: {len(forecast_df)} hours")
    return forecast_df


# ======================================================================
# Quick smoke-test when running standalone
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Pearls AQI Predictor — Inference Smoke Test")
    print("=" * 60)

    # Test weather forecast fetch (no Hopsworks key required)
    print("\n--- Weather Forecast Test ---")
    wf = get_weather_forecast(forecast_days=3)
    if not wf.empty:
        print(f"   Columns: {list(wf.columns[:8])} …")
        print(f"   Time range: {wf.index.min()} → {wf.index.max()}")
    else:
        print("   (empty — API may be unreachable)")

    print("\nSmoke test complete.  Full pipeline requires HOPSWORKS_API_KEY.")
