<![CDATA[<div align="center">

# 🌍 Pearls AQI Predictor

**Real-time Air Quality Index prediction for Karachi, Pakistan**

[![Feature Pipeline](https://github.com/<your-username>/DSPROJECTQ1/actions/workflows/feature_pipeline.yml/badge.svg)](https://github.com/<your-username>/DSPROJECTQ1/actions/workflows/feature_pipeline.yml)
[![Training Pipeline](https://github.com/<your-username>/DSPROJECTQ1/actions/workflows/training_pipeline.yml/badge.svg)](https://github.com/<your-username>/DSPROJECTQ1/actions/workflows/training_pipeline.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

*An end-to-end serverless ML system that ingests hourly air quality & weather data, engineers features, trains ensemble models, and serves 72-hour AQI forecasts through an interactive Streamlit dashboard.*

---

</div>

## 📖 Overview

Pearls AQI Predictor is a **production-grade MLOps pipeline** that continuously monitors and predicts the US EPA Air Quality Index (AQI) for Karachi. The system:

- **Ingests** hourly air-quality and meteorological data from the [Open-Meteo](https://open-meteo.com/) APIs
- **Engineers** 60+ temporal, lag, and rolling-window features
- **Trains** an ensemble of Random Forest, Gradient Boosting, and XGBoost regressors
- **Stores** features and models in [Hopsworks](https://www.hopsworks.ai/) Feature Store & Model Registry
- **Serves** real-time predictions and 72-hour forecasts via a [Streamlit](https://streamlit.io/) dashboard
- **Automates** everything with GitHub Actions — zero manual intervention

---

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph Data Sources
        A["🌐 Open-Meteo\nAir Quality API"]
        B["🌤️ Open-Meteo\nWeather API"]
    end

    subgraph Feature Pipeline — Hourly
        C["📥 Data Ingestion"]
        D["⚙️ Feature Engineering"]
    end

    subgraph Hopsworks
        E["📦 Feature Store"]
        F["🗂️ Model Registry"]
    end

    subgraph Training Pipeline — Daily
        G["🧠 Model Training\n(RF + GBR + XGB)"]
        H["📊 Evaluation &\nModel Selection"]
    end

    subgraph Serving
        I["🖥️ Streamlit\nDashboard"]
    end

    A --> C
    B --> C
    C --> D
    D --> E
    E --> G
    G --> H
    H --> F
    F --> I
    E --> I
```

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| 📡 | **Hourly Data Ingestion** | Automated hourly fetches from Open-Meteo APIs via GitHub Actions |
| 🔧 | **Advanced Feature Engineering** | 60+ features: lag (1–24h), rolling stats (6h/24h), cyclical time encodings, wind chill, pollution index |
| 🤖 | **Ensemble ML Models** | Random Forest, Gradient Boosting, and XGBoost — best model auto-selected |
| 📦 | **Hopsworks Feature Store** | Centralized, versioned feature storage with point-in-time correctness |
| 🔄 | **Automated Retraining** | Daily retraining via GitHub Actions keeps the model fresh |
| 📊 | **Interactive Dashboard** | Streamlit app with real-time AQI, 72-hour forecasts, and SHAP explanations |
| 🎨 | **EPA Color Coding** | AQI categories with official US EPA color scheme |
| 🚨 | **Health Alerts** | Automatic alerts when AQI exceeds unhealthy thresholds (>150) |
| 📈 | **SHAP Explainability** | Understand *why* the model predicts what it predicts |
| 🌐 | **Fully Serverless** | No infrastructure to manage — GitHub Actions + Hopsworks + Streamlit Cloud |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Data Source** | [Open-Meteo API](https://open-meteo.com/) | Air quality & weather data (free, no key required) |
| **Feature Store** | [Hopsworks](https://www.hopsworks.ai/) | Feature storage, versioning & model registry |
| **ML Framework** | Scikit-learn · XGBoost | Ensemble model training & evaluation |
| **Explainability** | SHAP | Model interpretability & feature importance |
| **Dashboard** | Streamlit · Plotly | Interactive web UI & visualizations |
| **Orchestration** | GitHub Actions | Scheduled pipeline execution (hourly / daily) |
| **Language** | Python 3.11 | Core development language |
| **Data Processing** | Pandas · NumPy | Data manipulation & numerical computation |

---

## 📁 Project Structure

```
DSPROJECTQ1/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # Hourly: ingest data → engineer features → Hopsworks
│       └── training_pipeline.yml   # Daily: train models → evaluate → register best model
├── .streamlit/
│   └── secrets.toml                # Streamlit secrets (not committed)
├── src/
│   ├── __init__.py                 # Package init
│   ├── config.py                   # Central configuration & constants
│   ├── backfill_pipeline.py        # Historical data backfill (90 days)
│   ├── feature_pipeline.py         # Data ingestion + feature engineering
│   ├── training_pipeline.py        # Model training + evaluation + registration
│   └── app.py                      # Streamlit dashboard
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
└── README.md                       # You are here!
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.11+** installed
- A free **[Hopsworks](https://app.hopsworks.ai/)** account
- **Git** installed

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/DSPROJECTQ1.git
cd DSPROJECTQ1
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the project root (or export directly):

```bash
# .env
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
```

Or export in your shell:

```bash
export HOPSWORKS_API_KEY="your_hopsworks_api_key_here"
```

> 💡 **Tip:** Get your API key from [Hopsworks App → Account Settings → API Keys](https://app.hopsworks.ai/)

### 5. Run the Backfill Pipeline

Fetch the last 90 days of historical data and populate the feature store:

```bash
python src/feature_pipeline.py
```

### 6. Train the Models

Train and evaluate ensemble models, then register the best performer:

```bash
python src/training_pipeline.py
```

### 7. Launch the Dashboard

```bash
streamlit run src/app.py
```

The dashboard will open at `http://localhost:8501` 🎉

---

## ⚙️ GitHub Actions Setup

### 1. Add Repository Secret

1. Go to your repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `HOPSWORKS_API_KEY`
4. Value: *your Hopsworks API key*

### 2. Enable GitHub Actions

1. Go to your repo → **Actions** tab
2. Click **"I understand my workflows, go ahead and enable them"**

### 3. Workflow Schedule

| Workflow | Schedule | Trigger |
|---|---|---|
| **Feature Pipeline** | Every hour (`0 * * * *`) | `schedule` + `workflow_dispatch` |
| **Training Pipeline** | Daily at midnight UTC (`0 0 * * *`) | `schedule` + `workflow_dispatch` |

> 📝 **Note:** You can manually trigger any workflow from the **Actions** tab → select workflow → **Run workflow**.

---

## 🌐 API Reference

This project uses the **free, open-source** [Open-Meteo API](https://open-meteo.com/) — no API key required.

| Endpoint | URL | Purpose |
|---|---|---|
| **Air Quality** | `air-quality-api.open-meteo.com/v1/air-quality` | PM2.5, PM10, CO, NO₂, SO₂, O₃, US AQI |
| **Historical Weather** | `archive-api.open-meteo.com/v1/archive` | Temp, humidity, wind, pressure, radiation |
| **Weather Forecast** | `api.open-meteo.com/v1/forecast` | Forecast weather for AQI prediction |

### Air Quality Parameters

`pm2_5` · `pm10` · `carbon_monoxide` · `nitrogen_dioxide` · `sulphur_dioxide` · `ozone` · `us_aqi` · `us_aqi_pm2_5` · `us_aqi_pm10` · `us_aqi_nitrogen_dioxide` · `us_aqi_carbon_monoxide` · `us_aqi_ozone` · `us_aqi_sulphur_dioxide`

### Weather Parameters

`temperature_2m` · `relative_humidity_2m` · `dewpoint_2m` · `apparent_temperature` · `pressure_msl` · `surface_pressure` · `windspeed_10m` · `winddirection_10m` · `windgusts_10m` · `precipitation` · `cloudcover` · `shortwave_radiation`

---

## 🤖 ML Models

### Algorithms

| Model | Estimators | Max Depth | Learning Rate |
|---|---|---|---|
| **Random Forest** | 200 | 12 | — |
| **Gradient Boosting** | 200 | 12 | 0.1 |
| **XGBoost** | 300 | 8 | 0.1 |

### Evaluation Metrics

All models are evaluated on a held-out 20% temporal test split using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination
- **MAPE** — Mean Absolute Percentage Error

The best-performing model is automatically selected and registered in the Hopsworks Model Registry.

### Feature Categories

| Category | Count | Examples |
|---|---|---|
| Raw Weather | 12 | Temperature, humidity, wind speed, pressure |
| Raw Air Quality | 12 | PM2.5, PM10, CO, NO₂, SO₂, O₃ (+ sub-AQIs) |
| Lag Features | 24 | `us_aqi_lag_1h` … `us_aqi_lag_24h` |
| Time Features | 11 | Hour, day-of-week, cyclical sin/cos encodings |
| Derived Features | 7 | Rolling mean/std (6h, 24h), wind chill, pollution index |

---

## 🖥️ Dashboard Preview

> 📸 *Screenshot coming soon — run `streamlit run src/app.py` to see it live!*

<!-- ![Dashboard Preview](docs/dashboard_preview.png) -->

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Guidelines

- Follow existing code style and conventions
- Add docstrings to new functions
- Update `config.py` for any new parameters
- Test locally before submitting a PR

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by Team Pearls**

*Karachi, Pakistan — Breathing cleaner, one prediction at a time.*

</div>
]]>
