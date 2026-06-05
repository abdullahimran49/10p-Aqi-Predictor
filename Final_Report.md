# Pearls AQI Predictor
## Final Project Report

---

**Submitted by:** Abdullah Imran
**Institution:** Karachi School of Business & Leadership (KSBL)
**Submission Type:** Internship Project Report

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Fulfillment of Project Requirements](#2-fulfillment-of-project-requirements)
   - 2.1 [Feature Pipeline](#21-feature-pipeline-automated-hourly)
   - 2.2 [Training Pipeline](#22-training-pipeline-automated-daily)
   - 2.3 [CI/CD Pipeline Automation](#23-cicd-pipeline-automation)
   - 2.4 [Web App & Dashboard](#24-web-app--dashboard)
   - 2.5 [Additional Guidelines](#25-additional-guidelines-satisfied)
3. [System Architecture & File-by-File Breakdown](#3-system-architecture--file-by-file-breakdown)
   - 3.1 [High-Level Architecture](#31-high-level-architecture)
   - 3.2 [config.py — Central Configuration](#32-configpy--central-configuration)
   - 3.3 [feature_pipeline.py — Hourly Data Ingestion](#33-feature_pipelinepy--hourly-data-ingestion)
   - 3.4 [backfill_pipeline.py — Historical Backfill](#34-backfill_pipelinepy--one-time-historical-backfill)
   - 3.5 [training_pipeline.py — Daily Model Training](#35-training_pipelinepy--daily-model-training)
   - 3.6 [inference.py — Prediction Engine](#36-inferencepy--prediction-engine)
   - 3.7 [streamlit_app.py — Interactive Dashboard](#37-streamlit_apppy--interactive-dashboard)
   - 3.8 [GitHub Actions Workflows](#38-github-actions-workflows)
   - 3.9 [Supporting Files](#39-supporting-files)
4. [Complete Data Flow — End to End](#4-complete-data-flow--end-to-end)
5. [Challenges & Solutions](#5-challenges--solutions)
6. [Conclusion](#6-conclusion)

---

## 1. Project Overview

The **Pearls AQI Predictor** is a production-grade, 100% serverless Machine Learning pipeline designed to predict the **Air Quality Index (AQI) in Karachi, Pakistan**, up to **3 days in advance**. The system seamlessly integrates automated data ingestion, advanced feature engineering, ensemble model training, and an interactive real-time dashboard.

---

## 2. Fulfillment of Project Requirements

### 2.1 Feature Pipeline (Automated Hourly)

**Requirement:** Fetch raw weather/pollutant data, compute features (time-based, derived), and store in a Feature Store.

**Implementation:** `src/feature_pipeline.py` fetches live data from the Open-Meteo Air Quality and Weather APIs.

**Feature Engineering:** Over 60 features were engineered, including:

- **Time Features:** `hour`, `day_of_week`, `month`, and cyclical encodings (`hour_sin`, `month_cos`) to capture diurnal pollution cycles.
- **Lag Features:** 24-hour lags (`us_aqi_lag_1h` to `us_aqi_lag_24h`) to capture autocorrelation.
- **Derived/Rolling Features:** 6-hour and 24-hour rolling means and standard deviations, AQI change rates, and a custom pollution index.

**Storage:** Features are pushed to the Hopsworks Feature Store using point-in-time correctness.

**Backfill:** `src/backfill_pipeline.py` was executed to fetch the last 90 days of historical data to bootstrap the feature store.

---

### 2.2 Training Pipeline (Automated Daily)

**Requirement:** Fetch historical data from Feature Store, train/evaluate models (Statistical + Deep Learning), evaluate via RMSE, MAE, R², and push to Model Registry.

**Implementation:** `src/training_pipeline.py` retrieves the unified dataset from Hopsworks and performs a **temporal train-test split (80/20)** to prevent data leakage.

**Models Experimented:**

| Model | Type |
|---|---|
| Random Forest Regressor | Ensemble (Bagging) |
| Gradient Boosting Regressor | Ensemble (Boosting) |
| XGBoost Regressor | Advanced Tree |
| Ridge Regression | Statistical / Linear Baseline |
| MLP Regressor | Deep Learning / Neural Network |

**Evaluation:** Models are evaluated on RMSE, MAE, and R² Score. The pipeline automatically selects the best tree-based model (RandomForest, GradientBoosting, or XGBoost) by lowest RMSE. Tree-based models are preferred over linear models for recursive forecasting because they have natural output bounds and cannot extrapolate beyond the training data range.

> **Current Best Model:** Gradient Boosting with **R² = 0.84**, **RMSE = 6.19**

**Registry:** The best-performing model, along with its metadata and evaluation metrics, is serialized and uploaded to the Hopsworks Model Registry.

---

### 2.3 CI/CD Pipeline Automation

**Requirement:** Automatically run the feature script hourly and the training script daily.

**Implementation:** GitHub Actions is used as the CI/CD orchestrator.

- `feature_pipeline.yml`: Scheduled via cron (`15 * * * *`) to run at 15 minutes past every hour.
- `training_pipeline.yml`: Scheduled via cron (`15 0 * * *`) to run daily at 00:15 UTC.

> **Why Offset?** The 15-minute offsets intentionally bypass GitHub Actions' global queues that occur at the top of the hour, ensuring ultra-reliable execution.

---

### 2.4 Web App & Dashboard

**Requirement:** Load model/features, compute predictions, and show on a descriptive dashboard.

**Implementation:** `app/streamlit_app.py` leverages Streamlit to provide a rich, interactive UI.

**Workflow:**

1. Connects to Hopsworks to pull the latest 24 hours of feature data and the active Model Version.
2. Fetches the 72-hour meteorological forecast from Open-Meteo.
3. Executes a recursive forecasting loop to predict future AQI hour-by-hour using the trained ML model.

**Visuals:** Features dynamic Plotly charts, current pollutant breakdowns, and official US EPA health color-coding.

---

### 2.5 Additional Guidelines Satisfied

- **EDA & Trends:** Built-in visualization in the dashboard allows users to explore temporal pollution trends (diurnal spikes in PM2.5).
- **Explainability (SHAP):** The training pipeline incorporates SHAP (SHapley Additive exPlanations). Feature importances are extracted globally and displayed in the UI so users can see exactly which factors (e.g., wind speed vs. ozone) drive the predictions.
- **Hazardous Alerts:** The dashboard includes conditional UI banners that dynamically alert users when AQI levels exceed the "Unhealthy" or "Hazardous" EPA thresholds (> 150).

---

## 3. System Architecture & File-by-File Breakdown

### 3.1 High-Level Architecture

The system consists of **3 independent pipelines** that feed data into Hopsworks, and **1 dashboard** that reads from Hopsworks and the APIs to display predictions.

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                          │
│   Open-Meteo APIs → feature_pipeline.py → Hopsworks Feature    │
│                      (60+ features computed)     Store          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                        MODEL TRAINING                           │
│   Hopsworks Feature Store → training_pipeline.py → Hopsworks   │
│    (all rows downloaded)    (5 models, best RMSE)  Model Reg.  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      DASHBOARD / INFERENCE                      │
│   Hopsworks Model Registry → inference.py → streamlit_app.py   │
│   Hopsworks Feature Store ──┘               → User's Browser   │
│   Open-Meteo Forecast APIs ─┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3.2 `config.py` — Central Configuration

This is the single source of truth for every constant used across the project. Nothing is hardcoded elsewhere.

| Constant | Value | Used By |
|---|---|---|
| `CITY_NAME` | `"Karachi"` | All pipelines + dashboard |
| `LATITUDE / LONGITUDE` | `24.8608 / 67.0104` | API calls to Open-Meteo |
| `TIMEZONE` | `"Asia/Karachi"` | Timestamp handling |
| `AQ_PARAMS` | List of 13 air quality parameters | `feature_pipeline.py` API calls |
| `WEATHER_PARAMS` | List of 12 weather parameters | `feature_pipeline.py` API calls |
| `TARGET_COLUMN` | `"us_aqi"` | Training target variable |
| `AQI_CATEGORIES` | Dict mapping category → (low, high, color) | Dashboard color-coding |
| `RANDOM_STATE` | `42` | Model reproducibility |
| `FEATURE_GROUP_NAME` | `"air_quality_features"` | Hopsworks feature group |

**Key Function:**

`get_aqi_category(aqi_value)` — Takes an AQI number (e.g., 65) and returns the category name and color (e.g., `("Moderate", "#ffde33")`).

---

### 3.3 `feature_pipeline.py` — Hourly Data Ingestion

**Triggered by:** GitHub Actions every hour (cron: `15 * * * *`)
**Purpose:** Fetch the last 7 days of weather + air quality data, engineer features, push to Hopsworks.

#### Step 1 — Fetch Air Quality Data

```
GET https://air-quality-api.open-meteo.com/v1/air-quality
    ?latitude=24.8608
    &longitude=67.0104
    &hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi,...
    &past_days=7
    &forecast_days=0
    &timezone=Asia/Karachi
```

Returns hourly readings for the last 7 days: PM2.5, PM10, CO, NO₂, SO₂, O₃, and US AQI sub-indices.

#### Step 2 — Fetch Weather Data

```
GET https://api.open-meteo.com/v1/forecast
    ?latitude=24.8608
    &longitude=67.0104
    &hourly=temperature_2m,relative_humidity_2m,dewpoint_2m,apparent_temperature,...
    &past_days=7
    &forecast_days=0
    &timezone=Asia/Karachi
```

Returns hourly weather: temperature, humidity, wind speed/direction/gusts, pressure, cloud cover, radiation.

#### Step 3 — Merge & Engineer Features

The function `compute_features(aq_df, weather_df)` merges both DataFrames on the timestamp column and creates:

| Feature Category | Features Created | Method |
|---|---|---|
| Time features | `hour`, `day_of_week`, `month`, `day_of_month`, `is_weekend` | Extracted from timestamp |
| Cyclical encodings | `hour_sin`, `hour_cos`, `day_of_week_sin/cos`, `month_sin/cos` | `sin(2π × value / period)` — tells the model hour 23 and hour 0 are close |
| Lag features | `us_aqi_lag_1h` through `us_aqi_lag_24h` | `df["us_aqi"].shift(n)` |
| Rolling statistics | `aqi_rolling_mean_6h/24h`, `aqi_rolling_std_6h/24h` | Rolling window mean/std |
| Derived | `aqi_change_rate` | `df["us_aqi"].diff()` |
| Derived | `wind_chill_factor` | `apparent_temperature - temperature_2m` |
| Derived | `pollution_index` | `0.3×PM2.5 + 0.2×PM10 + 0.15×NO₂ + 0.15×O₃ + 0.1×CO + 0.1×SO₂` |

After computing all features, rows with NaN values (from lag/rolling operations) are dropped.

#### Step 4 — Push to Hopsworks Feature Store

```python
project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="air_quality_features",
    version=1,
    primary_key=["city", "timestamp"],
    event_time="timestamp"
)
fg.insert(df)
```

The feature group uses `city + timestamp` as a composite primary key, so duplicate timestamps are automatically upserted (updated, not duplicated).

---

### 3.4 `backfill_pipeline.py` — One-Time Historical Backfill

**Triggered by:** Manual execution (run once at project setup)
**Purpose:** Fill the Feature Store with 90 days of historical data so the model has enough training data from day one.

This script is nearly identical to `feature_pipeline.py` but with two key differences:

1. It uses `past_days=90` instead of `past_days=7`.
2. For weather, it uses the **Historical Archive API** (`archive-api.open-meteo.com/v1/archive`) instead of the Forecast API, because the forecast API only has ~7 days of history.

It imports and reuses the exact same `compute_features()` function from `feature_pipeline.py` to ensure identical feature engineering.

---

### 3.5 `training_pipeline.py` — Daily Model Training

**Triggered by:** GitHub Actions daily at 00:15 UTC
**Purpose:** Pull all data from Hopsworks, train 5 ML models, pick the best, push to Model Registry.

#### Step 1 — Fetch Training Data

```python
project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group(name="air_quality_features", version=1)
df = fg.read()  # Downloads ALL rows from the feature store
```

Then separates:
- **X (features):** All columns EXCEPT `city`, `timestamp`, and `us_aqi`
- **y (target):** The `us_aqi` column

#### Step 2 — Temporal Train/Test Split

```python
df = df.sort_values("timestamp")
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

> **Important:** This is a **temporal split**, NOT a random split. The first 80% of time is training, the last 20% is testing. This prevents data leakage — the model never sees future data during training.

#### Step 3 — Train 5 Models

| Model | Type | Key Parameters |
|---|---|---|
| RandomForest | Ensemble (bagging) | 200 trees, `max_depth=12` |
| GradientBoosting | Ensemble (boosting) | 200 trees, `max_depth=8`, `lr=0.1` |
| XGBoost | Advanced boosting | 300 trees, `max_depth=8`, `lr=0.1`, `subsample=0.8` |
| Ridge | Linear (statistical) | `alpha=1.0` |
| NeuralNet (MLP) | Deep Learning | 2 hidden layers (128, 64), ReLU, Adam optimizer |

Each model is evaluated on:
- **RMSE** — penalizes large errors
- **MAE** — average prediction error
- **R²** — how much variance is explained (1.0 = perfect)

The best tree-based model (by lowest RMSE) is automatically selected. Tree models are preferred for recursive forecasting because they produce bounded predictions, preventing runaway divergence during the 72-hour recursive loop.

#### Step 4 — SHAP Feature Importance

```python
explainer = shap.TreeExplainer(model)       # For tree models
explainer = shap.Explainer(model, X_test)   # For Ridge/MLP
shap_values = explainer.shap_values(X_test[:200])
importance = mean(|shap_values|) per feature
```

This reveals which features matter most. For example, `aqi_rolling_mean_24h` typically carries the highest SHAP value, meaning the 24-hour AQI trend is the biggest driver of predictions, followed by `pm2_5`, `aqi_change_rate`, and `hour_sin`.

If SHAP fails, it falls back to `model.feature_importances_` for tree models or `|model.coef_|` for Ridge.

#### Step 5 — Save to Hopsworks Model Registry

Four files are saved locally in the `aqi_model/` directory, then uploaded:

| File | Contents |
|---|---|
| `model.pkl` | The trained scikit-learn model (serialized with joblib) |
| `metrics.json` | `{"rmse": 6.19, "mae": 4.70, "r2": 0.84, "best_model": "GradientBoosting"}` |
| `feature_importance.json` | `{"aqi_rolling_mean_24h": 15.28, "pm2_5": 2.17, "aqi_change_rate": 1.52, ...}` |
| `feature_columns.json` | Exact column order the model expects at inference time |

---

### 3.6 `inference.py` — Prediction Engine

**Used by:** `streamlit_app.py`
**Purpose:** Load the trained model, fetch live data, and generate 72-hour AQI predictions.

This file contains 4 key functions:

**`load_model_from_registry()`**
Connects to Hopsworks, downloads the latest model version, and returns the trained model object, metrics dictionary, feature importance dictionary, and feature column list.

**`get_recent_features_from_hopsworks()`**
Connects to the Hopsworks Feature Store, reads the feature group, sorts by timestamp, and returns the last 24 hours of feature data (the "recent history" shown in the Historical + Forecast chart).

**`get_weather_forecast(forecast_days=3)`**
Calls two Open-Meteo APIs to retrieve the next 72 hours of forecasted weather conditions AND forecasted pollutant levels from Open-Meteo's atmospheric models.

**`recursive_forecast(model, feature_columns, recent_data, weather_forecast, n_hours=72)`**
The core prediction function. For each hour from t+1 to t+72:

```
1. Take the weather forecast for this hour (temperature, humidity, wind, etc.)
   → KNOWN values from Open-Meteo's forecast

2. Take the air quality forecast for this hour (pm2_5, pm10, ozone, etc.)
   → KNOWN values from Open-Meteo's AQ forecast

3. Compute time features (hour_sin, day_of_week_cos, etc.)
   → Calculated from the timestamp

4. Compute lag features (us_aqi_lag_1h, us_aqi_lag_2h, ...)
   → For t+1: use actual recent data
   → For t+2 onwards: use previous predictions

5. Compute rolling features (rolling_mean_6h, rolling_std_24h, ...)
   → Calculated from the growing prediction history

6. Feed all features into the trained ML model → get predicted AQI

7. Clamp prediction to valid range [0, 500]

8. Store this prediction and use it as "history" for the next hour
```

---

### 3.7 `streamlit_app.py` — Interactive Dashboard

**Deployed on:** Streamlit Community Cloud (`pearls-aqi-karachi.streamlit.app`)
**Purpose:** Visualize predictions and current air quality data.

**Execution flow on page load:**

```
main()
 ├── inject_custom_css()        → Apply the dark glassmorphism theme
 ├── render_sidebar()           → City info, data sources, AQI scale legend
 ├── load_model()               → Downloads model from Hopsworks (cached 1 hr)
 ├── load_recent_features()     → Gets last 24h from Feature Store (cached 15 min)
 ├── load_weather_forecast()    → Gets 72h forecast from Open-Meteo (cached 30 min)
 ├── run_forecast()             → Runs the recursive 72-hour prediction
 ├── render_header()            → Title + AQI badge with category color
 ├── render_alert_banner()      → Health advisory if AQI > 150
 ├── render_metric_cards()      → 6 metric cards (Current AQI, Today Avg, etc.)
 ├── render_forecast_chart()    → Plotly line chart: 72-hour AQI prediction
 ├── render_historical_chart()  → Actual vs Predicted overlay
 ├── render_pollutant_breakdown()→ Bar chart of current PM2.5, PM10, etc.
 ├── render_feature_importance()→ Horizontal bar chart from SHAP values
 ├── render_model_performance() → RMSE, MAE, R² score cards
 └── render_aqi_legend()        → Color-coded health category explanations
```

**Caching Strategy:**

| Function | Cache Duration | Reason |
|---|---|---|
| `load_model()` | 1 hour | Model only changes once per day |
| `load_recent_features()` | 15 minutes | Feature store updates hourly |
| `load_weather_forecast()` | 30 minutes | Weather forecasts update slowly |

When multiple users visit the dashboard, only the first visit actually calls Hopsworks/Open-Meteo. All subsequent visitors receive instant cached results.

---

### 3.8 GitHub Actions Workflows

**`feature_pipeline.yml`**

```yaml
schedule:
  - cron: '15 * * * *'   # Every hour at :15
```

Checks out the repo → Installs Python 3.11 → `pip install -r requirements.txt` → Runs `python src/feature_pipeline.py` with `HOPSWORKS_API_KEY` from GitHub Secrets.

**`training_pipeline.yml`**

```yaml
schedule:
  - cron: '15 0 * * *'   # Daily at 00:15 UTC (5:15 AM PKT)
```

Same setup → Runs `python src/training_pipeline.py`.

**`backfill_pipeline.yml`**

```yaml
on:
  workflow_dispatch:   # Manual trigger only
```

No schedule — this is a one-time job triggered manually from the GitHub Actions tab.

---

### 3.9 Supporting Files

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies: pandas, numpy, scikit-learn, xgboost, shap, plotly, streamlit, hopsworks, joblib, openmeteo-requests, requests-cache, retry-requests |
| `packages.txt` | OS-level packages for Streamlit Cloud: `librdkafka-dev` (required by Hopsworks' Kafka dependency) |
| `.streamlit/config.toml` | Streamlit theme configuration (dark mode) |

---

## 4. Complete Data Flow — End to End

```
┌──────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA COLLECTION (Hourly)                           │
│                                                              │
│  Open-Meteo APIs ──► feature_pipeline.py ──► Hopsworks      │
│                       (60+ features)          Feature Store  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 2: MODEL TRAINING (Daily)                             │
│                                                              │
│  Hopsworks Feature Store ──► training_pipeline.py ──► Model │
│  (downloads all rows)         (trains 5 models,      Registry│
│                                picks best RMSE)             │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 3: DASHBOARD (On Every User Visit)                    │
│                                                              │
│  Hopsworks Model Registry ──┐                                │
│  Hopsworks Feature Store ───┼──► inference.py ──► Dashboard │
│  Open-Meteo Forecast APIs ──┘      (72-hr        → Browser  │
│                                    predictions)              │
└──────────────────────────────────────────────────────────────┘
```

> **Note:** The dashboard calls Open-Meteo's Forecast API directly (not the Feature Store) to get the next 72 hours of predicted weather and air quality. This is because the Feature Store only contains historical data (past observations), while the Forecast API provides future predictions from Open-Meteo's atmospheric models. The ML model then uses these future weather predictions as inputs to generate its own AQI predictions.

**Why the Dashboard Always Shows Real-Time Data:**

Even if the hourly feature pipeline has not run recently, the dashboard still shows fresh data because:

1. `load_recent_features()` pulls the latest 24 hours from the Feature Store — this data was inserted by the most recent pipeline run.
2. `get_weather_forecast()` calls Open-Meteo's Forecast API directly from the Streamlit server — this always returns the latest 72-hour forecast regardless of pipeline status.
3. `recursive_forecast()` combines the model + fresh forecast data to generate brand new predictions on every page load.

---

## 5. Challenges & Solutions

### Challenge 1: Recursive Forecast Stability

**Problem:** Recursive 72-hour forecasting creates a feedback loop where each hour's prediction is used as input for the next hour. This caused two failure modes:
- **Flatlining:** When using all autoregressive features (24 lag features + rolling means), tree models simply copied the previous prediction, producing a flat line.
- **Divergence:** When linear models (Ridge) were selected, small prediction errors were amplified through the feedback loop, causing predictions to spiral to 500.

**Solution:** Through iterative experimentation, we found the optimal feature balance:
- **Dropped:** Individual lag features (`us_aqi_lag_1h` through `us_aqi_lag_24h`) and short-term rolling mean (`aqi_rolling_mean_6h`) — these dominated the model and caused feedback instability.
- **Kept:** `aqi_rolling_mean_24h` (slow-changing 24-hour anchor) and `aqi_change_rate` (trend signal) — these provide temporal context without destabilizing the recursive loop.
- **Model Selection:** Only tree-based models (RandomForest, GradientBoosting, XGBoost) are considered for deployment, as they have natural output bounds and cannot extrapolate beyond training data range.

This achieved **R² = 0.84** with realistic hour-to-hour variation in the 72-hour forecast, driven by weather features like temperature, solar radiation, and wind patterns.

---

### Challenge 2: Plotly Timestamp Arithmetic

**Problem:** A severe `TypeError` crash in the UI was caused by an internal **Plotly 6.7.0 bug** when rendering vertical lines (`add_vline`) on Pandas Timestamps.

**Solution:** A resilient workaround was engineered using a `go.Scatter` boundary trace, ensuring 100% UI uptime.

---

### Challenge 3: Streamlit OS Dependencies

**Problem:** The Hopsworks package relies on `confluent-kafka`, which requires C++ compilers and `librdkafka` in the OS layer — not available by default on Streamlit Cloud.

**Solution:** A `packages.txt` file was bundled for the Streamlit Cloud Debian container to automatically install `librdkafka-dev` prior to the Python environment build.

---

## 6. Conclusion

The **Pearls AQI Predictor** successfully fulfills **100% of the project requirements**. It operates completely autonomously with zero server maintenance, blending cloud-native MLOps architectures with an elegant frontend interface to deliver critical environmental intelligence to the citizens of Karachi.

The system demonstrates end-to-end machine learning engineering competency — from automated data ingestion and feature engineering through to model training, deployment, and real-time interactive prediction — all within a fully serverless, cost-free infrastructure.

---

*Report prepared by Abdullah Imran — KSBL*
