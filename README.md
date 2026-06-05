# 🌍 Pearls AQI Predictor

**Real-Time Air Quality Index (AQI) Prediction for Karachi, Pakistan**

[![Feature Pipeline](https://github.com/abdullahimran49/10p-Aqi-Predictor/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/abdullahimran49/10p-Aqi-Predictor/actions/workflows/feature_pipeline.yml)
[![Training Pipeline](https://github.com/abdullahimran49/10p-Aqi-Predictor/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/abdullahimran49/10p-Aqi-Predictor/actions/workflows/training_pipeline.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Hopsworks](https://img.shields.io/badge/MLOps-Hopsworks-00B4D8)

**🌐 Live Dashboard:** [https://pearls-aqi-karachi.streamlit.app/](https://pearls-aqi-karachi.streamlit.app/)

---

## 📖 Project Overview

The **Pearls AQI Predictor** is a production-grade, **100% serverless** end-to-end Machine Learning pipeline that predicts the **US EPA Air Quality Index (AQI)** for Karachi, Pakistan up to **72 hours (3 days) in advance**.

The system runs fully autonomously with zero human intervention:
- **Every hour:** A GitHub Actions workflow fetches live weather and pollutant data, engineers 60+ features, and pushes them to the Hopsworks Feature Store.
- **Every day:** A separate workflow trains 5 ML models, evaluates them on RMSE/MAE/R², selects the best, and uploads it to the Hopsworks Model Registry.
- **On every visit:** The Streamlit dashboard loads the latest model and features, fetches the 72-hour weather forecast, and runs a recursive ML prediction loop.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA COLLECTION (Hourly)            │
│                                                                 │
│   Open-Meteo APIs ──► feature_pipeline.py ──► Hopsworks        │
│   (AQ + Weather)       (60+ features)          Feature Store    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: MODEL TRAINING (Daily)              │
│                                                                 │
│   Hopsworks Feature Store ──► training_pipeline.py ──► Model   │
│   (downloads all rows)        (5 models trained,       Registry │
│                                best RMSE selected)              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 3: DASHBOARD (On User Visit)           │
│                                                                 │
│   Hopsworks Model Registry ──┐                                  │
│   Hopsworks Feature Store ───┼──► inference.py ──► Dashboard    │
│   Open-Meteo Forecast APIs ──┘    (72-hour          → Browser  │
│                                    recursive ML                 │
│                                    predictions)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|---|---|
| **Data Sources** | Open-Meteo Air Quality API, Open-Meteo Weather Forecast API |
| **Feature Store** | Hopsworks (serverless, free tier) |
| **Model Registry** | Hopsworks Model Registry |
| **Orchestration / CI/CD** | GitHub Actions (cron-triggered workflows) |
| **ML Frameworks** | Scikit-Learn, XGBoost, SHAP |
| **Deep Learning** | Scikit-Learn MLPRegressor (Neural Network) |
| **Dashboard** | Streamlit Community Cloud |
| **Visualization** | Plotly |
| **Language** | Python 3.11 |

---

## 📂 Project Structure

```
10p-Aqi-Predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml      # Hourly cron (15 * * * *)
│   ├── training_pipeline.yml     # Daily cron  (15 0 * * *)
│   └── backfill_pipeline.yml     # Manual trigger (one-time)
├── app/
│   └── streamlit_app.py          # Interactive dashboard (Streamlit)
├── src/
│   ├── config.py                 # Central configuration (all constants)
│   ├── feature_pipeline.py       # Hourly: fetch data → engineer features → push to Hopsworks
│   ├── backfill_pipeline.py      # One-time: 90-day historical data backfill
│   ├── training_pipeline.py      # Daily: train 5 models → select best → push to registry
│   └── inference.py              # Prediction engine: 72-hour recursive ML forecast
├── Final_Report.md               # Detailed project report
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── packages.txt                  # OS-level dependencies for Streamlit Cloud
```

---

## 🔬 Feature Engineering

The feature pipeline engineers **60+ features** from raw API data:

| Category | Features | Method |
|---|---|---|
| **Raw Pollutants** (6) | `pm2_5`, `pm10`, `carbon_monoxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone` | Direct from Open-Meteo AQ API |
| **Raw Weather** (12) | `temperature_2m`, `relative_humidity_2m`, `dewpoint_2m`, `apparent_temperature`, `pressure_msl`, `surface_pressure`, `windspeed_10m`, `winddirection_10m`, `windgusts_10m`, `precipitation`, `cloudcover`, `shortwave_radiation` | Direct from Open-Meteo Weather API |
| **Time Features** (5) | `hour`, `day_of_week`, `month`, `day_of_month`, `is_weekend` | Extracted from timestamp |
| **Cyclical Encodings** (6) | `hour_sin/cos`, `day_of_week_sin/cos`, `month_sin/cos` | `sin(2π × value / period)` — ensures hour 23 and hour 0 are treated as adjacent |
| **Lag Features** (24) | `us_aqi_lag_1h` through `us_aqi_lag_24h` | `df["us_aqi"].shift(n)` — captures autocorrelation |
| **Rolling Statistics** (4) | `aqi_rolling_mean_6h/24h`, `aqi_rolling_std_6h/24h` | Rolling window mean and standard deviation |
| **Derived** (3) | `aqi_change_rate`, `wind_chill_factor`, `pollution_index` | `diff()`, `apparent_temp - temp`, weighted pollutant sum |

> **Note:** Not all engineered features are used for training. The training pipeline carefully selects a subset optimized for recursive 72-hour forecasting stability (see [Challenges](#-challenges--solutions)).

---

## 🤖 Model Training & Evaluation

### Models Trained Daily

| Model | Type | Category | Key Hyperparameters |
|---|---|---|---|
| **Random Forest** | Ensemble (Bagging) | Scikit-Learn | 200 trees, `max_depth=12` |
| **Gradient Boosting** | Ensemble (Boosting) | Scikit-Learn | 200 trees, `max_depth=8`, `lr=0.1` |
| **XGBoost** | Advanced Boosting | XGBoost Library | 300 trees, `max_depth=8`, `lr=0.1`, `subsample=0.8` |
| **Ridge Regression** | Linear / Statistical | Scikit-Learn | `alpha=1.0` |
| **MLP Neural Network** | Deep Learning | Scikit-Learn | 2 hidden layers (128, 64), ReLU, Adam, 500 epochs |

### Evaluation Metrics

All models are evaluated using a **temporal train/test split (80/20)** — the first 80% of time is used for training, the last 20% for testing. This prevents data leakage (the model never sees future data during training).

| Metric | Description |
|---|---|
| **RMSE** (Root Mean Squared Error) | Penalizes large prediction errors more heavily |
| **MAE** (Mean Absolute Error) | Average absolute prediction error |
| **R²** (Coefficient of Determination) | Fraction of variance explained (1.0 = perfect) |

### Model Selection Strategy

The pipeline selects the **tree-based model with the lowest RMSE** (from RandomForest, GradientBoosting, XGBoost). Tree-based models are preferred for recursive forecasting because they have **natural output bounds** — they can only predict values within the range of training data, preventing runaway divergence in the 72-hour recursive loop.

### Current Performance

| Metric | Value |
|---|---|
| **Best Model** | Gradient Boosting |
| **R² Score** | 0.84 |
| **RMSE** | 6.19 |
| **MAE** | 4.70 |

### SHAP Feature Importance

The training pipeline uses **SHAP (SHapley Additive exPlanations)** to compute feature importance for the best model. This reveals which factors drive AQI predictions:

| Feature | SHAP Importance | Interpretation |
|---|---|---|
| `aqi_rolling_mean_24h` | 15.28 | 24-hour AQI trend is the strongest predictor |
| `pm2_5` | 2.17 | PM2.5 concentration directly affects AQI |
| `aqi_change_rate` | 1.52 | Whether AQI is trending up or down |
| `hour_sin` | 1.33 | Time-of-day effects (diurnal pollution cycle) |
| `pm10` | 0.96 | Coarse particulate matter concentration |

---

## 🔄 CI/CD Pipeline Automation

### GitHub Actions Workflows

| Workflow | Schedule | Purpose |
|---|---|---|
| **Feature Pipeline** | `15 * * * *` (every hour at :15) | Fetch data → engineer features → push to Hopsworks |
| **Training Pipeline** | `15 0 * * *` (daily at 00:15 UTC) | Train 5 models → select best → push to Model Registry |
| **Backfill Pipeline** | Manual trigger only | One-time 90-day historical data load |

> **Why :15 offset?** GitHub Actions cron jobs scheduled at the top of the hour (`:00`) experience severe delays due to global queue congestion. The 15-minute offset ensures reliable execution.

Both pipelines authenticate with Hopsworks securely via **GitHub Repository Secrets** (`HOPSWORKS_API_KEY`).

---

## 📊 Interactive Dashboard

The Streamlit dashboard (`app/streamlit_app.py`) provides:

| Section | Description |
|---|---|
| **AQI Badge** | Current AQI with US EPA category color-coding |
| **Metric Cards** | Current AQI, Today's Average, Tomorrow's Average, 72h Max, PM2.5, Temperature |
| **3-Day Forecast Chart** | Plotly line chart with hourly AQI predictions for the next 72 hours |
| **Historical + Forecast** | Overlay of actual historical data and predicted future values |
| **Pollutant Breakdown** | Bar chart showing current levels of PM2.5, PM10, O₃, NO₂, CO, SO₂ |
| **Feature Importance** | Horizontal bar chart from SHAP analysis |
| **Model Performance** | RMSE, MAE, R² score cards with the selected model name |
| **Health Alerts** | Conditional warning banner when AQI exceeds 150 (Unhealthy threshold) |
| **AQI Legend** | US EPA health category explanations with color coding |

### Caching Strategy

| Data | Cache Duration | Reason |
|---|---|---|
| Model from Hopsworks | 1 hour | Model only changes once per day |
| Recent features | 15 minutes | Feature store updates hourly |
| Weather forecast | 30 minutes | Weather forecasts update slowly |

---

## ⚡ Recursive Forecasting Engine

The `inference.py` module implements a **recursive single-step forecasting loop** for 72-hour AQI prediction:

```
For each hour t+1 to t+72:
  1. Get weather forecast for this hour     → from Open-Meteo Forecast API (known values)
  2. Get pollutant forecast for this hour    → from Open-Meteo AQ API (known values)
  3. Compute time features                   → from timestamp (hour_sin, day_of_week, etc.)
  4. Compute rolling mean from history       → from running prediction series
  5. Compute change rate                     → current - previous prediction
  6. Build feature vector → feed to ML model → get predicted AQI
  7. Clamp to valid range [0, 500]
  8. Append to history for next iteration
```

The forecast shows realistic hour-to-hour variation because weather features (temperature, solar radiation, wind speed) change throughout the day, creating natural diurnal AQI patterns.

---

## 🧩 Challenges & Solutions

### 1. Recursive Forecast Stability

**Problem:** Recursive forecasting creates a feedback loop — each prediction becomes input for the next. This caused two failure modes:
- **Flatlining:** Individual lag features (`us_aqi_lag_1h..24h`) made tree models copy the previous prediction, producing a flat line.
- **Divergence:** Linear models (Ridge) amplified small errors through the feedback loop, causing predictions to spiral to 500.

**Solution:** Careful feature selection — dropped individual lags and short-term rolling means, kept only `aqi_rolling_mean_24h` (slow-changing anchor) and `aqi_change_rate` (trend signal). Combined with tree-based model selection (bounded outputs), this produces stable forecasts with realistic variation.

### 2. Plotly Timestamp Bug

**Problem:** Plotly 6.7.0 crashed with `TypeError` when using `add_vline()` with Pandas Timestamps.

**Solution:** Replaced with `go.Scatter` boundary traces for 100% UI stability.

### 3. Streamlit Cloud Dependencies

**Problem:** Hopsworks requires `confluent-kafka` which needs `librdkafka-dev` (C++ library) not available on Streamlit Cloud by default.

**Solution:** Added `packages.txt` with `librdkafka-dev` for automatic OS-level installation.

---

## ⚙️ Running Locally

### Prerequisites
- Python 3.11
- [Hopsworks API Key](https://app.hopsworks.ai/)

### Setup

```bash
git clone https://github.com/abdullahimran49/10p-Aqi-Predictor.git
cd 10p-Aqi-Predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Hopsworks API key
export HOPSWORKS_API_KEY="your_api_key_here"   # Windows: set HOPSWORKS_API_KEY=your_key
```

### Run Pipelines

```bash
# 1. Backfill 90 days of historical data (first time only)
python src/backfill_pipeline.py

# 2. Run feature pipeline (fetches latest data)
python src/feature_pipeline.py

# 3. Train models and register best
python src/training_pipeline.py

# 4. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## 📄 Documentation

- **[Final_Report.md](Final_Report.md)** — Comprehensive project report documenting architecture, implementation details, feature engineering, model evaluation, challenges, and solutions.

---

**Submitted by:** Abdullah Imran — Karachi School of Business & Leadership (KSBL)
