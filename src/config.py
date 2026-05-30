"""
Pearls AQI Predictor — Central Configuration
=============================================
All project-wide constants, API URLs, feature lists, and Hopsworks config.
"""

# ──────────────────────────────────────────────
# City Configuration
# ──────────────────────────────────────────────
CITY_NAME = "Karachi"
CITY_COUNTRY = "Pakistan"
LATITUDE = 24.8608
LONGITUDE = 67.0104
TIMEZONE = "Asia/Karachi"

# ──────────────────────────────────────────────
# Open-Meteo API Endpoints
# ──────────────────────────────────────────────
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HISTORICAL_WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# ──────────────────────────────────────────────
# Air Quality Parameters (for Open-Meteo API)
# ──────────────────────────────────────────────
AQ_PARAMS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
    "us_aqi_pm2_5",
    "us_aqi_pm10",
    "us_aqi_nitrogen_dioxide",
    "us_aqi_carbon_monoxide",
    "us_aqi_ozone",
    "us_aqi_sulphur_dioxide",
]

# ──────────────────────────────────────────────
# Weather Parameters (for Open-Meteo API)
# ──────────────────────────────────────────────
WEATHER_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "windspeed_10m",
    "winddirection_10m",
    "windgusts_10m",
    "precipitation",
    "cloudcover",
    "shortwave_radiation",
]

# ──────────────────────────────────────────────
# Feature Engineering Configuration
# ──────────────────────────────────────────────
TARGET_COLUMN = "us_aqi"
LAG_HOURS = list(range(1, 25))  # 1h to 24h lags
ROLLING_WINDOWS = [6, 24]       # 6-hour and 24-hour rolling windows
BACKFILL_DAYS = 90              # Historical backfill period

# Feature columns that the model will use (excluding target)
# These are built dynamically during feature engineering
TIME_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]

DERIVED_FEATURES = [
    "aqi_change_rate",
    "aqi_rolling_mean_6h",
    "aqi_rolling_mean_24h",
    "aqi_rolling_std_6h",
    "aqi_rolling_std_24h",
    "wind_chill_factor",
    "pollution_index",
]

# ──────────────────────────────────────────────
# Hopsworks Configuration
# ──────────────────────────────────────────────
HOPSWORKS_PROJECT_NAME = None  # Auto-detected from API key
FEATURE_GROUP_NAME = "air_quality_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "aqi_predictor"
MODEL_VERSION = 1

# ──────────────────────────────────────────────
# AQI Categories (US EPA Standard)
# ──────────────────────────────────────────────
AQI_CATEGORIES = {
    "Good":                        (0, 50,    "#00e400"),
    "Moderate":                    (51, 100,  "#ffff00"),
    "Unhealthy for Sensitive":     (101, 150, "#ff7e00"),
    "Unhealthy":                   (151, 200, "#ff0000"),
    "Very Unhealthy":              (201, 300, "#8f3f97"),
    "Hazardous":                   (301, 500, "#7e0023"),
}

AQI_ALERT_THRESHOLD = 150  # Alert when AQI exceeds this

# ──────────────────────────────────────────────
# Model Training Configuration
# ──────────────────────────────────────────────
TEST_SIZE = 0.2           # 80/20 temporal split
RANDOM_STATE = 42
N_ESTIMATORS_RF = 200     # Random Forest
N_ESTIMATORS_GBR = 200    # Gradient Boosting
MAX_DEPTH = 12
LEARNING_RATE_GBR = 0.1
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# ──────────────────────────────────────────────
# Forecast Configuration
# ──────────────────────────────────────────────
FORECAST_HOURS = 72       # 3 days × 24 hours


def get_aqi_category(aqi_value):
    """Return the AQI category name and color for a given AQI value."""
    for category, (low, high, color) in AQI_CATEGORIES.items():
        if low <= aqi_value <= high:
            return category, color
    return "Hazardous", "#7e0023"


def get_feature_columns():
    """Return the full list of feature columns used by the model."""
    # Raw weather features
    weather_feats = WEATHER_PARAMS.copy()

    # Raw AQ features (excluding the target us_aqi)
    aq_feats = [p for p in AQ_PARAMS if p != TARGET_COLUMN]

    # Lag features
    lag_feats = [f"us_aqi_lag_{h}h" for h in LAG_HOURS]

    # Time + derived features
    all_features = weather_feats + aq_feats + lag_feats + TIME_FEATURES + DERIVED_FEATURES

    return all_features
