"""
Pearls AQI Predictor — Streamlit Dashboard
==========================================
A stunning, production-quality air quality forecasting dashboard.
"""

# ── Bootstrap: env vars & path setup (BEFORE any src/ imports) ──────────
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st  # noqa: E402

# Inject HOPSWORKS_API_KEY from Streamlit secrets if not already set
try:
    if "HOPSWORKS_API_KEY" not in os.environ:
        os.environ["HOPSWORKS_API_KEY"] = st.secrets["HOPSWORKS_API_KEY"]
except Exception:
    pass

# ── Standard imports ────────────────────────────────────────────────────
from datetime import datetime, timedelta  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from src.config import (  # noqa: E402
    AQI_ALERT_THRESHOLD,
    AQI_CATEGORIES,
    CITY_COUNTRY,
    CITY_NAME,
    LATITUDE,
    LONGITUDE,
    TARGET_COLUMN,
    TIMEZONE,
    get_aqi_category,
)

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium Dark Glassmorphism Theme
# ═════════════════════════════════════════════════════════════════════════
def inject_custom_css():
    st.markdown(
        """
    <style>
    /* ── Google Font ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global ──────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1040 40%, #0d1933 70%, #0a0e27 100%);
    }

    /* ── Scrollbar ───────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
    ::-webkit-scrollbar-thumb { background: rgba(120,90,220,0.4); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(120,90,220,0.7); }

    /* ── Glassmorphism Card ──────────────────────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 28px 32px;
        margin-bottom: 20px;
        animation: fadeInUp 0.6s ease-out forwards;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }

    /* ── Metric Cards ────────────────────────────── */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        padding: 22px 20px;
        text-align: center;
        animation: fadeInUp 0.7s ease-out forwards;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        border-color: rgba(255,255,255,0.15);
    }
    .metric-card .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 6px 0 4px 0;
    }
    .metric-card .metric-label {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .metric-card .metric-delta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* ── AQI Badge ───────────────────────────────── */
    .aqi-badge {
        display: inline-flex;
        align-items: center;
        gap: 14px;
        padding: 16px 36px;
        border-radius: 50px;
        font-weight: 800;
        font-size: 1.3rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.3);
        animation: pulse-glow 2s ease-in-out infinite;
    }
    .aqi-badge .aqi-number {
        font-size: 2.8rem;
        font-weight: 900;
        line-height: 1;
    }

    /* ── Header ──────────────────────────────────── */
    .main-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        animation: fadeInUp 0.5s ease-out forwards;
        line-height: 1.2;
    }
    .subtitle {
        font-size: 1.15rem;
        color: rgba(255,255,255,0.45);
        font-weight: 400;
        margin-top: 2px;
        animation: fadeInUp 0.6s ease-out forwards;
    }

    /* ── Alert Banner ────────────────────────────── */
    .alert-banner {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 16px;
        padding: 20px 28px;
        margin: 16px 0;
        animation: fadeInUp 0.6s ease-out forwards;
    }
    .alert-banner h3 {
        color: #fca5a5;
        margin: 0 0 8px 0;
        font-size: 1.1rem;
    }
    .alert-banner p {
        color: rgba(255,255,255,0.7);
        margin: 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* ── Section Title ───────────────────────────── */
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: rgba(255,255,255,0.9);
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(120,90,220,0.3);
        display: inline-block;
    }

    /* ── AQI Legend ───────────────────────────────── */
    .aqi-legend-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 8px;
        background: rgba(255,255,255,0.04);
        margin: 4px;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.8);
        transition: background 0.2s ease;
    }
    .aqi-legend-item:hover {
        background: rgba(255,255,255,0.08);
    }
    .aqi-legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Sidebar ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,10,40,0.97) 0%, rgba(10,14,39,0.97) 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: rgba(255,255,255,0.85) !important;
    }
    .sidebar-info {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.06);
        font-size: 0.85rem;
        color: rgba(255,255,255,0.65);
        line-height: 1.7;
    }

    /* ── Loading State ───────────────────────────── */
    .loading-state {
        text-align: center;
        padding: 60px 30px;
        animation: fadeInUp 0.5s ease-out forwards;
    }
    .loading-state .emoji {
        font-size: 4rem;
        margin-bottom: 16px;
    }
    .loading-state h2 {
        color: rgba(255,255,255,0.85);
        margin-bottom: 10px;
    }
    .loading-state p {
        color: rgba(255,255,255,0.5);
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* ── Animations ───────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 4px 25px rgba(0,0,0,0.3); }
        50%      { box-shadow: 0 4px 35px rgba(0,0,0,0.5); }
    }

    /* ── Hide Streamlit defaults ─────────────────── */
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Plotly chart containers ──────────────────── */
    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


inject_custom_css()


# ═════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═════════════════════════════════════════════════════════════════════════
@st.cache_resource(ttl=3600, show_spinner=False)
def load_model():
    """Load model, metrics, feature importance, and feature columns from Hopsworks."""
    from src.inference import load_model_from_registry

    return load_model_from_registry()


@st.cache_data(ttl=900, show_spinner=False)
def load_recent_features():
    """Load recent feature data from the Hopsworks feature store."""
    from src.inference import get_recent_features_from_hopsworks

    return get_recent_features_from_hopsworks()


@st.cache_data(ttl=1800, show_spinner=False)
def load_weather_forecast():
    """Load upcoming weather + AQ forecast data."""
    from src.inference import get_weather_forecast

    return get_weather_forecast(forecast_days=3)


def run_forecast(model, feature_columns, recent_data, weather_forecast, n_hours=72):
    """Run recursive AQI forecast."""
    from src.inference import recursive_forecast

    return recursive_forecast(model, feature_columns, recent_data, weather_forecast, n_hours)


# ═════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════

def create_plotly_layout(title="", height=450, showlegend=True):
    """Return a consistent dark-themed Plotly layout."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, font=dict(size=18, color="rgba(255,255,255,0.85)")),
        font=dict(family="Inter, sans-serif", color="rgba(255,255,255,0.7)"),
        height=height,
        showlegend=showlegend,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=50, r=30, t=60, b=40),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.05)",
        ),
    )


def add_aqi_bands(fig, y_max=350):
    """Add coloured AQI category bands to a Plotly figure."""
    for cat_name, (low, high, color) in AQI_CATEGORIES.items():
        band_high = min(high, y_max)
        if low > y_max:
            continue
        fig.add_hrect(
            y0=low,
            y1=band_high,
            fillcolor=color,
            opacity=0.07,
            layer="below",
            line_width=0,
            annotation_text=cat_name if band_high - low > 20 else "",
            annotation_position="top left",
            annotation=dict(font_size=9, font_color=color, opacity=0.6),
        )


def metric_card_html(label, value, color="#a78bfa", delta=None, delta_color="#4ade80"):
    """Return HTML for a styled metric card."""
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "●"
        d_color = "#ef4444" if delta > 0 else "#4ade80" if delta < 0 else "rgba(255,255,255,0.5)"
        delta_html = f'<div class="metric-delta" style="color:{d_color}">{arrow} {abs(delta):.1f}</div>'
    return f"""
    <div class="metric-card" style="border-left: 3px solid {color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        {delta_html}
    </div>
    """


# ═════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌍 Pearls AQI")
        st.markdown("---")

        # City info
        st.markdown(
            f"""
        <div class="sidebar-info">
            <strong>📍 City:</strong> {CITY_NAME}, {CITY_COUNTRY}<br/>
            <strong>🌐 Coordinates:</strong> {LATITUDE}°N, {LONGITUDE}°E<br/>
            <strong>🕐 Timezone:</strong> {TIMEZONE}
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Data sources
        st.markdown("### 📡 Data Sources")
        st.markdown(
            """
        <div class="sidebar-info">
            <strong>Weather:</strong> Open-Meteo API<br/>
            <strong>Air Quality:</strong> Open-Meteo AQ API<br/>
            <strong>Feature Store:</strong> Hopsworks<br/>
            <strong>Model Registry:</strong> Hopsworks
        </div>
        """,
            unsafe_allow_html=True,
        )

        # AQI Scale
        st.markdown("### 🎨 AQI Scale")
        legend_html = ""
        for cat_name, (low, high, color) in AQI_CATEGORIES.items():
            legend_html += f"""
            <div class="aqi-legend-item">
                <div class="aqi-legend-dot" style="background:{color}"></div>
                {cat_name} ({low}–{high})
            </div>
            """
        st.markdown(legend_html, unsafe_allow_html=True)

        st.markdown("---")

        # Refresh
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.markdown(
            "<div style='text-align:center; color:rgba(255,255,255,0.25); font-size:0.75rem; margin-top:30px;'>"
            "Built with ❤️ by Pearls Team</div>",
            unsafe_allow_html=True,
        )


render_sidebar()


# ═════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ═════════════════════════════════════════════════════════════════════════

def render_header(current_aqi=None):
    """Render the top header with title and AQI badge."""
    st.markdown('<div class="main-title">🌍 Pearls AQI Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="subtitle">Real-time Air Quality Forecasting for {CITY_NAME}, {CITY_COUNTRY}</div>',
        unsafe_allow_html=True,
    )

    if current_aqi is not None:
        cat_name, cat_color = get_aqi_category(current_aqi)
        st.markdown(
            f"""
        <div style="margin-top:18px;">
            <div class="aqi-badge" style="background:rgba(0,0,0,0.35); border:2px solid {cat_color}; color:{cat_color};">
                <span class="aqi-number">{int(current_aqi)}</span>
                <span>{cat_name}</span>
            </div>
            <span style="color:rgba(255,255,255,0.35); font-size:0.8rem; margin-left:16px;">
                Last updated: {datetime.now().strftime("%b %d, %Y  %I:%M %p")}
            </span>
        </div>
        """,
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


def render_alert_banner(current_aqi):
    """Show a health advisory alert if AQI exceeds threshold."""
    if current_aqi is None or current_aqi <= AQI_ALERT_THRESHOLD:
        return
    cat_name, cat_color = get_aqi_category(current_aqi)
    health_advisories = {
        "Unhealthy": "Everyone may begin to experience health effects. Members of sensitive groups may experience more serious effects. Reduce prolonged outdoor exertion.",
        "Very Unhealthy": "Health alert: everyone may experience more serious health effects. Avoid prolonged outdoor exertion. Keep windows closed.",
        "Hazardous": "Health warning of emergency conditions. The entire population is likely to be affected. Stay indoors and avoid all outdoor physical activity.",
    }
    advisory = health_advisories.get(cat_name, health_advisories["Unhealthy"])
    st.markdown(
        f"""
    <div class="alert-banner">
        <h3>⚠️ Air Quality Alert — {cat_name} (AQI {int(current_aqi)})</h3>
        <p>{advisory}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_metric_cards(forecast_df, recent_df):
    """Render key metric cards row."""
    current_aqi = forecast_df["predicted_aqi"].iloc[0] if len(forecast_df) > 0 else None

    # Today's avg / tomorrow's avg
    now = pd.Timestamp.now()
    today_mask = forecast_df["timestamp"].dt.date == now.date() if "timestamp" in forecast_df.columns else pd.Series([False] * len(forecast_df))
    tomorrow_mask = forecast_df["timestamp"].dt.date == (now + timedelta(days=1)).date() if "timestamp" in forecast_df.columns else pd.Series([False] * len(forecast_df))

    today_avg = forecast_df.loc[today_mask, "predicted_aqi"].mean() if today_mask.any() else current_aqi
    tomorrow_avg = forecast_df.loc[tomorrow_mask, "predicted_aqi"].mean() if tomorrow_mask.any() else None
    max_pred = forecast_df["predicted_aqi"].max() if len(forecast_df) > 0 else None

    # Pollutant levels from recent data
    pm25 = recent_df["pm2_5"].iloc[-1] if "pm2_5" in recent_df.columns and len(recent_df) > 0 else None
    temp = recent_df["temperature_2m"].iloc[-1] if "temperature_2m" in recent_df.columns and len(recent_df) > 0 else None

    cols = st.columns(6)

    cards = [
        ("Current AQI", f"{int(current_aqi)}" if current_aqi else "—", get_aqi_category(current_aqi)[1] if current_aqi else "#a78bfa", None),
        ("Today Avg", f"{int(today_avg)}" if today_avg else "—", "#60a5fa", None),
        ("Tomorrow Avg", f"{int(tomorrow_avg)}" if tomorrow_avg else "—", "#818cf8", (tomorrow_avg - today_avg) if (today_avg and tomorrow_avg) else None),
        ("72h Max", f"{int(max_pred)}" if max_pred else "—", "#f472b6", None),
        ("PM2.5", f"{pm25:.1f}" if pm25 else "—", "#fb923c", None),
        ("Temperature", f"{temp:.1f}°C" if temp else "—", "#34d399", None),
    ]

    for i, (label, value, color, delta) in enumerate(cards):
        with cols[i]:
            st.markdown(metric_card_html(label, value, color, delta), unsafe_allow_html=True)


def render_forecast_chart(forecast_df):
    """Render the main 3-day AQI forecast chart."""
    st.markdown('<div class="section-title">📈 3-Day AQI Forecast</div>', unsafe_allow_html=True)

    fig = go.Figure()

    y_max = max(350, forecast_df["predicted_aqi"].max() * 1.1) if len(forecast_df) > 0 else 350
    add_aqi_bands(fig, y_max=y_max)

    # Main forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["predicted_aqi"],
            mode="lines",
            name="Predicted AQI",
            line=dict(color="#a78bfa", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.08)",
            hovertemplate="<b>%{x|%b %d, %I:%M %p}</b><br>AQI: %{y:.0f}<extra></extra>",
        )
    )

    # Colour the markers by category
    if "aqi_color" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["predicted_aqi"],
                mode="markers",
                name="AQI Level",
                marker=dict(color=forecast_df["aqi_color"], size=5, line=dict(width=0)),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Alert threshold line
    fig.add_hline(
        y=AQI_ALERT_THRESHOLD,
        line_dash="dot",
        line_color="rgba(239,68,68,0.5)",
        line_width=1.5,
        annotation_text=f"Alert Threshold ({AQI_ALERT_THRESHOLD})",
        annotation_position="top right",
        annotation=dict(font_size=10, font_color="rgba(239,68,68,0.7)"),
    )

    fig.update_layout(
        **create_plotly_layout("Hourly AQI Prediction — Next 72 Hours", height=480),
        xaxis_title="Time",
        yaxis_title="AQI (US EPA)",
        yaxis_range=[0, y_max],
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_historical_forecast_chart(recent_df, forecast_df):
    """Render combined historical + forecast chart."""
    st.markdown('<div class="section-title">🕰️ Historical + Forecast</div>', unsafe_allow_html=True)

    fig = go.Figure()

    y_vals = []

    # Historical actual AQI
    if TARGET_COLUMN in recent_df.columns and len(recent_df) > 0:
        ts_col = None
        for col in ["timestamp", "date", "datetime"]:
            if col in recent_df.columns:
                ts_col = col
                break
        if ts_col is None and recent_df.index.name == "timestamp":
            hist_ts = recent_df.index
        elif ts_col:
            hist_ts = recent_df[ts_col]
        else:
            hist_ts = pd.date_range(end=pd.Timestamp.now(), periods=len(recent_df), freq="h")

        hist_aqi = recent_df[TARGET_COLUMN]
        y_vals.extend(hist_aqi.dropna().tolist())

        fig.add_trace(
            go.Scatter(
                x=hist_ts,
                y=hist_aqi,
                mode="lines",
                name="Actual AQI",
                line=dict(color="#60a5fa", width=2.5),
                hovertemplate="<b>%{x|%b %d, %I:%M %p}</b><br>Actual AQI: %{y:.0f}<extra></extra>",
            )
        )

    # Forecast
    if len(forecast_df) > 0:
        y_vals.extend(forecast_df["predicted_aqi"].dropna().tolist())
        fig.add_trace(
            go.Scatter(
                x=forecast_df["timestamp"],
                y=forecast_df["predicted_aqi"],
                mode="lines",
                name="Predicted AQI",
                line=dict(color="#a78bfa", width=2.5, dash="dash"),
                hovertemplate="<b>%{x|%b %d, %I:%M %p}</b><br>Predicted AQI: %{y:.0f}<extra></extra>",
            )
        )

    y_max = max(350, max(y_vals) * 1.1) if y_vals else 350
    add_aqi_bands(fig, y_max=y_max)

    # Boundary line
    if len(forecast_df) > 0:
        boundary_time = forecast_df["timestamp"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[boundary_time, boundary_time],
                y=[0, y_max],
                mode="lines",
                name="Now",
                line=dict(color="rgba(255,255,255,0.5)", width=1.5, dash="dot"),
                hoverinfo="skip",
                showlegend=False
            )
        )

    fig.update_layout(
        **create_plotly_layout("Actual vs. Predicted AQI", height=430),
        xaxis_title="Time",
        yaxis_title="AQI (US EPA)",
        yaxis_range=[0, y_max],
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_pollutant_breakdown(recent_df):
    """Render current pollutant levels bar chart."""
    st.markdown('<div class="section-title">🧪 Current Pollutant Levels</div>', unsafe_allow_html=True)

    pollutants = {
        "PM2.5": "pm2_5",
        "PM10": "pm10",
        "CO": "carbon_monoxide",
        "NO₂": "nitrogen_dioxide",
        "SO₂": "sulphur_dioxide",
        "O₃": "ozone",
    }

    values = []
    labels = []
    for display_name, col in pollutants.items():
        if col in recent_df.columns and len(recent_df) > 0:
            val = recent_df[col].iloc[-1]
            if pd.notna(val):
                values.append(val)
                labels.append(display_name)

    if not values:
        st.info("Pollutant data not available yet.")
        return

    # Normalize for colour coding (0-1)
    max_val = max(values) if max(values) > 0 else 1
    normed = [v / max_val for v in values]
    colors = [
        f"rgba({int(50 + 200*n)}, {int(220 - 180*n)}, {int(100)}, 0.85)"
        for n in normed
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            hovertemplate="<b>%{x}</b><br>Level: %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        **create_plotly_layout("", height=380, showlegend=False),
        xaxis_title="",
        yaxis_title="Concentration",
        bargap=0.35,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_feature_importance(feature_importance):
    """Render top 15 most important features."""
    st.markdown('<div class="section-title">🧠 Feature Importance</div>', unsafe_allow_html=True)

    if not feature_importance:
        st.info("Feature importance data not available.")
        return

    # Sort and take top 15
    sorted_feats = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [f[0].replace("_", " ").title() for f in reversed(sorted_feats)]
    importances = [f[1] for f in reversed(sorted_feats)]

    # Gradient colours (purple to blue)
    n = len(names)
    colors = [
        f"rgba({int(80 + (167-80)*i/max(n-1,1))}, {int(120 + (139-120)*i/max(n-1,1))}, {int(200 + (250-200)*i/max(n-1,1))}, 0.85)"
        for i in range(n)
    ]

    fig = go.Figure(
        go.Bar(
            y=names,
            x=importances,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0), cornerradius=4),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        **create_plotly_layout("", height=max(400, n * 30), showlegend=False),
        xaxis_title="Importance Score",
        yaxis_title="",
        margin=dict(l=180, r=30, t=30, b=40),
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_model_performance(metrics):
    """Render model performance metrics."""
    st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)

    if not metrics:
        st.info("Model metrics not available yet.")
        return

    cols = st.columns(4)

    metric_display = [
        ("RMSE", metrics.get("rmse", metrics.get("RMSE")), "#ef4444", "lower is better"),
        ("MAE", metrics.get("mae", metrics.get("MAE")), "#fb923c", "lower is better"),
        ("R² Score", metrics.get("r2", metrics.get("R2", metrics.get("r2_score"))), "#4ade80", "closer to 1"),
        ("Model", metrics.get("best_model", metrics.get("model_type", "N/A")), "#a78bfa", "selected best"),
    ]

    for i, (label, value, color, helper) in enumerate(metric_display):
        with cols[i]:
            if isinstance(value, float):
                display_val = f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
            elif value is not None:
                display_val = str(value)
            else:
                display_val = "—"
            st.markdown(
                f"""
                <div class="metric-card" style="border-left:3px solid {color};">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color}; font-size:1.8rem;">{display_val}</div>
                    <div style="color:rgba(255,255,255,0.3); font-size:0.72rem; margin-top:4px;">{helper}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_aqi_legend():
    """Render the AQI category legend table."""
    st.markdown('<div class="section-title">🎨 AQI Categories & Health Implications</div>', unsafe_allow_html=True)

    health_info = {
        "Good": "Air quality is satisfactory. Air pollution poses little or no risk.",
        "Moderate": "Acceptable; however, some pollutants may be a concern for a very small number of people.",
        "Unhealthy for Sensitive": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
        "Unhealthy": "Everyone may begin to experience health effects; sensitive groups may experience more serious effects.",
        "Very Unhealthy": "Health alert: everyone may experience more serious health effects.",
        "Hazardous": "Health warning of emergency conditions. The entire population is more likely to be affected.",
    }

    cols = st.columns(3)
    for idx, (cat_name, (low, high, color)) in enumerate(AQI_CATEGORIES.items()):
        with cols[idx % 3]:
            st.markdown(
                f"""
            <div class="metric-card" style="border-top:3px solid {color}; text-align:left; padding:16px 18px;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                    <div style="width:14px;height:14px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                    <strong style="color:{color}; font-size:0.95rem;">{cat_name}</strong>
                    <span style="color:rgba(255,255,255,0.4); font-size:0.78rem;">({low}–{high})</span>
                </div>
                <div style="color:rgba(255,255,255,0.5); font-size:0.8rem; line-height:1.5;">
                    {health_info.get(cat_name, '')}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_no_data_state(message="Setting Up", detail=""):
    """Render a friendly 'no data yet' state."""
    st.markdown(
        f"""
    <div class="glass-card loading-state">
        <div class="emoji">🔧</div>
        <h2>{message}</h2>
        <p>{detail}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════

def main():
    # ── Check API key is available ──
    if "HOPSWORKS_API_KEY" not in os.environ:
        render_header()
        render_no_data_state(
            "🔑 API Key Required",
            "Please set <code>HOPSWORKS_API_KEY</code> in your environment variables or in "
            "<code>.streamlit/secrets.toml</code> to connect to the feature store.",
        )
        return

    # ── Load data with graceful error handling ──
    model = None
    metrics = {}
    feature_importance = {}
    feature_columns = []
    recent_df = pd.DataFrame()
    forecast_df = pd.DataFrame()

    # 1. Load model
    model_loaded = False
    try:
        with st.spinner("Loading model from Hopsworks..."):
            model, metrics, feature_importance, feature_columns = load_model()
        model_loaded = True
    except Exception as e:
        st.warning(f"⚠️ Could not load model: {e}")

    # 2. Load recent features
    recent_loaded = False
    try:
        with st.spinner("Fetching recent air quality data..."):
            recent_df = load_recent_features()
        if recent_df is not None and len(recent_df) > 0:
            recent_loaded = True
    except Exception as e:
        st.warning(f"⚠️ Could not load recent data: {e}")

    # 3. Run forecast
    forecast_available = False
    if model_loaded and recent_loaded:
        try:
            with st.spinner("Generating 72-hour forecast..."):
                weather_forecast = load_weather_forecast()
                forecast_df = run_forecast(model, feature_columns, recent_df, weather_forecast, n_hours=72)
            if forecast_df is not None and len(forecast_df) > 0:
                forecast_available = True
        except Exception as e:
            st.warning(f"⚠️ Could not generate forecast: {e}")

    # Ensure DataFrames
    if recent_df is None:
        recent_df = pd.DataFrame()
    if forecast_df is None:
        forecast_df = pd.DataFrame()

    # ── Determine current AQI ──
    current_aqi = None
    if forecast_available and len(forecast_df) > 0:
        current_aqi = forecast_df["predicted_aqi"].iloc[0]
    elif recent_loaded and TARGET_COLUMN in recent_df.columns and len(recent_df) > 0:
        current_aqi = recent_df[TARGET_COLUMN].iloc[-1]

    # ── Render Dashboard ──
    render_header(current_aqi)

    # Alert banner
    if current_aqi is not None:
        render_alert_banner(current_aqi)

    # If no data at all, show setup instructions
    if not model_loaded and not recent_loaded:
        render_no_data_state(
            "Data Not Available Yet",
            "It looks like the feature pipeline hasn't run yet. Please run the data ingestion and training pipelines first, "
            "then come back here to see your AQI predictions. "
            "<br/><br/>"
            "Steps:<br/>"
            "1️⃣ Run <code>src/backfill.py</code> to ingest historical data<br/>"
            "2️⃣ Run <code>src/training.py</code> to train and register the model<br/>"
            "3️⃣ Refresh this dashboard 🎉",
        )
        render_aqi_legend()
        return

    # ── Key Metrics Row ──
    if forecast_available or recent_loaded:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        render_metric_cards(forecast_df, recent_df)
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── 3-Day Forecast Chart ──
    if forecast_available:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        render_forecast_chart(forecast_df)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Historical + Forecast ──
    if recent_loaded:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        render_historical_forecast_chart(recent_df, forecast_df)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Two-column: Pollutants + Feature Importance ──
    col_left, col_right = st.columns(2)

    with col_left:
        if recent_loaded:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            render_pollutant_breakdown(recent_df)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        if feature_importance:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            render_feature_importance(feature_importance)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Model Performance ──
    if metrics:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        render_model_performance(metrics)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── AQI Legend ──
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    render_aqi_legend()
    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
    <div style="text-align:center; color:rgba(255,255,255,0.2); font-size:0.78rem; margin-top:40px; padding-bottom:20px;">
        Pearls AQI Predictor • Powered by Open-Meteo, Hopsworks & Streamlit •
        Data refreshes every 15 minutes
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
else:
    main()
