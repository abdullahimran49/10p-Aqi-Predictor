"""
Pearls AQI Predictor — Backfill Pipeline (One-Shot)
====================================================
Run ONCE to hydrate the Hopsworks Feature Store with ~90 days of
historical air-quality and weather data for Karachi.

Re-uses all feature-engineering functions from ``feature_pipeline.py``
so there is zero logic duplication.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Path fix — guarantees `from src.…` imports work from any working directory.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests

from src.config import (
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    AIR_QUALITY_API_URL,
    HISTORICAL_WEATHER_API_URL,
    AQ_PARAMS,
    WEATHER_PARAMS,
    BACKFILL_DAYS,
)

from src.feature_pipeline import (
    fetch_air_quality_data,
    engineer_features,
    upload_to_hopsworks,
)

# ═══════════════════════════════════════════════════════════════════════════
# HISTORICAL WEATHER FETCHER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_historical_weather_data(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch hourly historical weather from the Open-Meteo Archive API.

    Parameters
    ----------
    start_date : str
        ISO date string, e.g. ``"2025-12-01"``.
    end_date : str
        ISO date string, e.g. ``"2026-02-28"``.

    Returns
    -------
    pd.DataFrame
        Hourly weather observations with a ``time`` column.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(WEATHER_PARAMS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
    }
    print(f"[WX-HIST] Fetching historical weather: {start_date} → {end_date} …")
    resp = requests.get(HISTORICAL_WEATHER_API_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    print(f"[WX-HIST] Retrieved {len(df)} hourly rows.")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def run_backfill_pipeline() -> None:
    """Execute the full backfill pipeline for BACKFILL_DAYS of history."""
    start = datetime.now(timezone.utc)
    print("=" * 60)
    print(f"  BACKFILL PIPELINE — started at {start.isoformat()}")
    print(f"  Loading {BACKFILL_DAYS} days of historical data")
    print("=" * 60)

    try:
        # --- Date range ---------------------------------------------------
        today = datetime.now(timezone.utc).date()
        start_date = (today - timedelta(days=BACKFILL_DAYS)).isoformat()
        end_date = today.isoformat()
        print(f"[INFO] Date range: {start_date} → {end_date}")

        # --- Step 1: Fetch raw data ---------------------------------------
        aq_df = fetch_air_quality_data(
            past_days=BACKFILL_DAYS,
            forecast_days=0,
        )
        wx_df = fetch_historical_weather_data(
            start_date=start_date,
            end_date=end_date,
        )

        # --- Step 2: Engineer features (same logic as hourly pipeline) ----
        df = engineer_features(aq_df, wx_df)

        # --- Summary statistics -------------------------------------------
        print("\n" + "─" * 60)
        print("  BACKFILL SUMMARY")
        print("─" * 60)
        print(f"  Final shape           : {df.shape}")
        print(f"  Columns               : {len(df.columns)}")
        print(f"  Timestamp range       : {df['timestamp'].min()} → "
              f"{df['timestamp'].max()}")
        print(f"  us_aqi  mean / std    : "
              f"{df['us_aqi'].mean():.1f} / {df['us_aqi'].std():.1f}")
        print(f"  us_aqi  min / max     : "
              f"{df['us_aqi'].min():.0f} / {df['us_aqi'].max():.0f}")
        print(f"  Missing values total  : {df.isna().sum().sum()}")
        print("─" * 60 + "\n")

        # --- Step 3: Upload to Hopsworks ---------------------------------
        upload_to_hopsworks(df)

    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] API request failed: {exc}")
        raise
    except Exception as exc:
        print(f"[ERROR] Backfill pipeline failed: {exc}")
        raise

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"  BACKFILL PIPELINE — finished in {elapsed:.1f}s")
    print(f"{'=' * 60}")


# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_backfill_pipeline()
