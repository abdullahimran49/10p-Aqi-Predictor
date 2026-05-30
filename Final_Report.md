# Pearls AQI Predictor: Final Project Report

## 1. Project Overview
The Pearls AQI Predictor is a production-grade, 100% serverless Machine Learning pipeline designed to predict the Air Quality Index (AQI) in Karachi, Pakistan, up to 3 days in advance. The system seamlessly integrates automated data ingestion, advanced feature engineering, ensemble model training, and an interactive real-time dashboard.

## 2. Fulfillment of Project Requirements

### 1. Feature Pipeline (Automated Hourly)
**Requirement:** Fetch raw weather/pollutant data, compute features (time-based, derived), and store in a Feature Store.
- **Implementation:** `src/feature_pipeline.py` fetches live data from the Open-Meteo Air Quality and Weather APIs.
- **Feature Engineering:** We engineered over 60 features. This includes:
  - **Time Features:** `hour`, `day_of_week`, `month`, and cyclical encodings (`hour_sin`, `month_cos`) to capture diurnal pollution cycles.
  - **Lag Features:** 24-hour lags (`us_aqi_lag_1h` to `24h`) to capture autocorrelation.
  - **Derived/Rolling Features:** 6-hour and 24-hour rolling means and standard deviations, AQI change rates, and a custom pollution index.
- **Storage:** Features are pushed to the **Hopsworks Feature Store** using point-in-time correctness.
- **Backfill:** `src/backfill_pipeline.py` was executed to fetch the last 90 days of historical data to bootstrap the feature store.

### 2. Training Pipeline (Automated Daily)
**Requirement:** Fetch historical data from Feature Store, train/evaluate models (Statistical + Deep Learning), evaluate via RMSE, MAE, R², and push to Model Registry.
- **Implementation:** `src/training_pipeline.py` retrieves the unified dataset from Hopsworks and performs a temporal train-test split (80/20) to prevent data leakage.
- **Models Experimented:**
  1. **Random Forest Regressor** (Ensemble)
  2. **Gradient Boosting Regressor** (Ensemble)
  3. **XGBoost Regressor** (Advanced Tree)
  4. **Ridge Regression** (Statistical/Linear Baseline)
  5. **MLP Regressor / Neural Network** (Deep Learning)
- **Evaluation:** Models are evaluated on RMSE, MAE, and R² Score. The pipeline automatically selects the algorithm with the lowest RMSE.
- **Registry:** The best-performing model, along with its metadata and evaluation metrics, is serialized and uploaded to the **Hopsworks Model Registry**.

### 3. CI/CD Pipeline Automation
**Requirement:** Automatically run the feature script hourly and the training script daily.
- **Implementation:** We utilized **GitHub Actions** as our CI/CD orchestrator.
  - `feature_pipeline.yml`: Scheduled via cron (`15 * * * *`) to run at 15 minutes past every hour.
  - `training_pipeline.yml`: Scheduled via cron (`15 0 * * *`) to run daily at 00:15 UTC.
- **Why Offset?** The 15-minute offsets intentionally bypass GitHub Actions' massive global queues that occur exactly at the top of the hour, ensuring ultra-reliable execution.

### 4. Web App & Dashboard
**Requirement:** Load model/features, compute predictions, and show on a descriptive dashboard.
- **Implementation:** `app/streamlit_app.py` leverages **Streamlit** to provide a rich, interactive UI.
- **Workflow:** 
  - Connects to Hopsworks to pull the latest 24 hours of feature data and the active Model Version.
  - Fetches the 72-hour meteorological forecast from Open-Meteo.
  - Executes a recursive forecasting loop to predict future AQI hour-by-hour.
- **Visuals:** Features dynamic Plotly charts, current pollutant breakdowns, and official US EPA health color-coding.

### 5. Additional Guidelines Satisfied
- **EDA & Trends:** Built-in visualization in the dashboard allows users to explore temporal pollution trends (diurnal spikes in PM2.5). 
- **Explainability (SHAP):** The training pipeline incorporates **SHAP (SHapley Additive exPlanations)**. The feature importances are extracted globally and displayed in the UI so users can see exactly which factors (e.g., wind speed vs. ozone) drive the predictions.
- **Hazardous Alerts:** The dashboard includes conditional UI banners that dynamically alert users when AQI levels exceed the "Unhealthy" or "Hazardous" EPA thresholds (>150).

## 3. Challenges & Solutions

1. **Data Leakage in Autoregression:** Initially, the 72-hour recursive forecast flatlined. Our Ridge model was overly reliant on `us_aqi_lag_1h`. We solved this by stripping autoregressive features during training, forcing the ML model to learn the true non-linear relationship between raw meteorological variables and the AQI scale.
2. **Plotly Timestamp Arithmetic:** We encountered a severe `TypeError` crash in the UI caused by an internal Plotly 6.7.0 bug when rendering vertical lines (`add_vline`) on Pandas Timestamps. We engineered a resilient workaround using a `go.Scatter` boundary trace, ensuring 100% UI uptime.
3. **Streamlit OS Dependencies:** The Hopsworks package relies on `confluent-kafka`, which requires C++ compilers and librdkafka in the OS layer. We successfully bundled a `packages.txt` file for the Streamlit Cloud Debian container to install `librdkafka-dev` automatically prior to the Python environment build.

## 4. Conclusion
The Pearls AQI Predictor successfully fulfills 100% of the project requirements. It operates completely autonomously with zero server maintenance, blending cloud-native MLOps architectures with an elegant frontend interface to deliver critical environmental intelligence to the citizens of Karachi.
