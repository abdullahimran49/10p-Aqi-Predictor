"""
Pearls AQI Predictor — Feature Pipeline (Hourly)
=================================================
Runs every hour via GitHub Actions to:
  1. Fetch the last 7 days of air-quality & weather data from Open-Meteo.
  2. Engineer time, derived, lag, and rolling features.
  3. Upsert the resulting DataFrame into the Hopsworks Feature Store.

Shared feature-engineering helpers are importable by backfill_pipeline.py.
"""

import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path fix — guarantees `from src.config import …` works regardless of cwd.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests
import hopsworks

from src.config import (
    CITY_NAME,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    AIR_QUALITY_API_URL,
    FORECAST_WEATHER_API_URL,
    AQ_PARAMS,
    WEATHER_PARAMS,
    TARGET_COLUMN,
    LAG_HOURS,
    ROLLING_WINDOWS,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════

def fetch_air_quality_data(
    past_days: int = 7,
    forecast_days: int = 0,
) -> pd.DataFrame:
    """Fetch hourly air-quality data from the Open-Meteo Air Quality API.

    Parameters
    ----------
    past_days : int
        Number of past days to retrieve.
    forecast_days : int
        Number of forecast days (0 for historical only).

    Returns
    -------
    pd.DataFrame
        Hourly air-quality observations indexed by ``time``.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(AQ_PARAMS),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": TIMEZONE,
    }
    print(f"[AQ] Fetching {past_days} days of air-quality data …")
    resp = requests.get(AIR_QUALITY_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    print(f"[AQ] Retrieved {len(df)} hourly rows.")
    return df


def fetch_weather_forecast_data(
    past_days: int = 7,
    forecast_days: int = 0,
) -> pd.DataFrame:
    """Fetch hourly weather data from the Open-Meteo Forecast API.

    Uses the *forecast* endpoint with ``past_days`` so it mirrors the
    air-quality window for the hourly feature pipeline.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_PARAMS),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": TIMEZONE,
    }
    print(f"[WX] Fetching {past_days} days of weather-forecast data …")
    resp = requests.get(FORECAST_WEATHER_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    print(f"[WX] Retrieved {len(df)} hourly rows.")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING  (shared with backfill_pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def merge_dataframes(
    aq_df: pd.DataFrame,
    wx_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join air-quality and weather DataFrames on ``time``."""
    merged = pd.merge(aq_df, wx_df, on="time", how="inner")
    print(f"[MERGE] Merged shape: {merged.shape}")
    return merged


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar & cyclical time features derived from ``time``."""
    ts = pd.to_datetime(df["time"])

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["day_of_month"] = ts.dt.day
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    print(f"[FEAT] Added time features.")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add AQI-derived, rolling, and composite features."""
    # Change rate
    df["aqi_change_rate"] = df[TARGET_COLUMN].diff()

    # Rolling statistics
    for window in ROLLING_WINDOWS:
        df[f"aqi_rolling_mean_{window}h"] = (
            df[TARGET_COLUMN].rolling(window=window, min_periods=1).mean()
        )
        df[f"aqi_rolling_std_{window}h"] = (
            df[TARGET_COLUMN].rolling(window=window, min_periods=1).std()
        )

    # Wind-chill factor
    df["wind_chill_factor"] = df["apparent_temperature"] - df["temperature_2m"]

    # Composite pollution index
    df["pollution_index"] = (
        0.3 * df["pm2_5"]
        + 0.2 * df["pm10"]
        + 0.15 * df["nitrogen_dioxide"]
        + 0.15 * df["ozone"]
        + 0.1 * df["carbon_monoxide"]
        + 0.1 * df["sulphur_dioxide"]
    )

    print(f"[FEAT] Added derived features.")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hourly lag features for us_aqi (1 h … 24 h)."""
    for lag in LAG_HOURS:
        df[f"us_aqi_lag_{lag}h"] = df[TARGET_COLUMN].shift(lag)
    print(f"[FEAT] Added {len(LAG_HOURS)} lag features (1h–24h).")
    return df


def finalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add metadata columns, rename index, and drop NaN rows."""
    df["city"] = CITY_NAME
    df = df.rename(columns={"time": "timestamp"})

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    print(f"[CLEAN] Dropped {before - after} NaN rows → {after} rows remain.")
    return df


def engineer_features(
    aq_df: pd.DataFrame,
    wx_df: pd.DataFrame,
) -> pd.DataFrame:
    """Full feature-engineering pipeline (merge → features → clean).

    This is the single entry-point used by both the hourly feature
    pipeline and the one-off backfill pipeline.
    """
    df = merge_dataframes(aq_df, wx_df)
    df = add_time_features(df)
    df = add_derived_features(df)
    df = add_lag_features(df)
    df = finalise_dataframe(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3.  HOPSWORKS UPLOAD
# ═══════════════════════════════════════════════════════════════════════════

def upload_to_hopsworks(df: pd.DataFrame) -> None:
    """Connect to Hopsworks and upsert ``df`` into the feature group.

    The ``HOPSWORKS_API_KEY`` environment variable must be set.
    """
    print("[HW] Logging into Hopsworks …")
    project = hopsworks.login()
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        description="Hourly air-quality features for Karachi AQI prediction.",
    )

    print(f"[HW] Inserting {len(df)} rows into feature group "
          f"'{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION} …")
    fg.insert(df, write_options={"wait_for_job": False})
    print("[HW] Insert triggered successfully (async).")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def run_feature_pipeline() -> None:
    """Execute the full hourly feature pipeline."""
    start = datetime.now(timezone.utc)
    print("=" * 60)
    print(f"  FEATURE PIPELINE — started at {start.isoformat()}")
    print("=" * 60)

    try:
        # --- Step 1: Fetch raw data --------------------------------
        aq_df = fetch_air_quality_data(past_days=7, forecast_days=0)
        wx_df = fetch_weather_forecast_data(past_days=7, forecast_days=0)

        # --- Step 2: Engineer features -----------------------------
        df = engineer_features(aq_df, wx_df)

        print(f"\n[INFO] Final DataFrame shape: {df.shape}")
        print(f"[INFO] Columns: {list(df.columns)}")
        print(f"[INFO] Timestamp range: {df['timestamp'].min()} → "
              f"{df['timestamp'].max()}")

        # --- Step 3: Upload to Hopsworks --------------------------
        upload_to_hopsworks(df)

    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] API request failed: {exc}")
        raise
    except Exception as exc:
        print(f"[ERROR] Feature pipeline failed: {exc}")
        raise

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"  FEATURE PIPELINE — finished in {elapsed:.1f}s")
    print(f"{'=' * 60}")


# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_feature_pipeline()
