# Pearls AQI Predictor

**Real-Time Air Quality Index (AQI) Prediction for Karachi, Pakistan**

[![Feature Pipeline](https://github.com/abdullahimran49/10p-Aqi-Predictor/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/abdullahimran49/10p-Aqi-Predictor/actions/workflows/feature_pipeline.yml)
[![Training Pipeline](https://github.com/abdullahimran49/10p-Aqi-Predictor/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/abdullahimran49/10p-Aqi-Predictor/actions/workflows/training_pipeline.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)

**🌐 Live Dashboard:** [https://pearls-aqi-karachi.streamlit.app/](https://pearls-aqi-karachi.streamlit.app/)

An end-to-end, serverless Machine Learning pipeline that continuously monitors meteorological data to predict the US EPA Air Quality Index (AQI) for Karachi up to 72 hours in advance.

---

## 📖 Project Overview

The **Pearls AQI Predictor** is designed to address the severe air pollution challenges in Karachi by providing accurate, real-time forecasts. The system operates as a fully automated MLOps pipeline:

1. **Automated Data Ingestion:** GitHub Actions automatically fetch live weather and pollutant data from the Open-Meteo API every hour.
2. **Feature Engineering & Storage:** Data is processed and pushed to a centralized Hopsworks Feature Store.
3. **Daily Model Retraining:** An ensemble of Machine Learning models (Random Forest, Gradient Boosting, Ridge) is evaluated daily to ensure the system adapts to shifting environmental trends.
4. **Interactive Dashboard:** A Streamlit web application serves the latest predictions alongside the US EPA health advisory guidelines.

## 🏗️ System Architecture

The project architecture relies on cloud-native, serverless technologies to ensure zero-maintenance operation:

- **Data Sources:** Open-Meteo Air Quality & Weather Forecast APIs.
- **Feature Store & Model Registry:** Hopsworks (Serverless ML data platform).
- **Orchestration:** GitHub Actions (Cron-triggered pipelines).
- **User Interface:** Streamlit Community Cloud.

## 🚀 Key Features

- **72-Hour Recursive Forecasting:** Accurately maps forecasted particulate matter (PM2.5, PM10) to the official US EPA AQI scale.
- **Real-Time Data Pipelines:** The `feature_pipeline` runs automatically every hour, ensuring the dashboard never goes stale.
- **Model Auto-Selection:** The `training_pipeline` runs daily, evaluating multiple algorithms and seamlessly deploying the model with the lowest Root Mean Square Error (RMSE).
- **SHAP Explainability:** The system utilizes SHAP (SHapley Additive exPlanations) to identify which meteorological factors (e.g., wind speed, temperature) are driving pollution levels.

## 🛠️ Technology Stack

- **Core Language:** Python 3.11
- **Machine Learning:** Scikit-Learn, XGBoost, SHAP
- **Data Engineering:** Pandas, NumPy
- **Cloud & MLOps:** Hopsworks, GitHub Actions
- **Data Visualization:** Plotly, Streamlit

---

## ⚙️ Running the Project Locally

To run the pipelines and the dashboard on your local machine, follow these steps:

### 1. Prerequisites
Ensure you have Python 3.11 installed and an active API key from [Hopsworks](https://app.hopsworks.ai/).

### 2. Setup Environment
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/abdullahimran49/10p-Aqi-Predictor.git
cd 10p-Aqi-Predictor

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Hopsworks
Set your Hopsworks API key as an environment variable:

```bash
# On Windows
set HOPSWORKS_API_KEY=your_api_key_here

# On macOS/Linux
export HOPSWORKS_API_KEY="your_api_key_here"
```

### 4. Execute the Pipelines
If you are running the project for the first time, backfill the historical data, then train the model:

```bash
# 1. Fetch historical data and populate the Feature Store
python src/feature_pipeline.py

# 2. Train the models and register the best performer
python src/training_pipeline.py
```

### 5. Launch the Dashboard
Run the Streamlit application to view the interactive UI:

```bash
streamlit run app/streamlit_app.py
```

---

## 🔄 CI/CD Automation

This repository utilizes **GitHub Actions** to fully automate the ML lifecycle. The workflow files are located in `.github/workflows/`:

- `feature_pipeline.yml`: Triggers hourly (`0 * * * *`) to fetch new data and update the Feature Store.
- `training_pipeline.yml`: Triggers daily at midnight UTC (`0 0 * * *`) to retrain the models on the newly accumulated data.

Both pipelines authenticate with Hopsworks securely via GitHub Repository Secrets (`HOPSWORKS_API_KEY`).

---

