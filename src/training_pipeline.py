"""
Pearls AQI Predictor — Daily Training Pipeline
================================================
Runs daily via GitHub Actions. Fetches features from Hopsworks, trains
multiple regression models, selects the best one by RMSE, computes SHAP
feature importances, and uploads the winning model to the Hopsworks
Model Registry.
"""

import os
import sys
import json
import shutil
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Make project root importable ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    LEARNING_RATE_GBR,
    MAX_DEPTH,
    MODEL_NAME,
    N_ESTIMATORS_GBR,
    N_ESTIMATORS_RF,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    XGBOOST_PARAMS,
)

warnings.filterwarnings("ignore")

# ======================================================================
# Step 1 — Fetch training data from Hopsworks
# ======================================================================

def fetch_training_data() -> pd.DataFrame:
    """Connect to Hopsworks and return the full feature-group DataFrame."""
    import hopsworks

    print("🔗  Connecting to Hopsworks …")
    project = hopsworks.login()

    print("📦  Fetching feature store …")
    fs = project.get_feature_store()

    print(f"📋  Reading feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION} …")
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()

    print(f"✅  Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Separate features (X) and target (y).

    Drops 'city', 'timestamp', and TARGET_COLUMN from X.
    Returns (X, y, feature_columns).
    """
    drop_cols = ["city", "timestamp", TARGET_COLUMN]
    existing_drop = [c for c in drop_cols if c in df.columns]

    y = df[TARGET_COLUMN].copy()
    X = df.drop(columns=existing_drop)
    
    # Drop data leakage columns (sub-AQI scores)
    leakage_cols = [c for c in X.columns if c.startswith("us_aqi_")]
    X = X.drop(columns=leakage_cols)
    
    feature_columns = list(X.columns)

    print(f"🎯  Target: {TARGET_COLUMN}  |  Features: {len(feature_columns)}")
    return X, y, feature_columns


# ======================================================================
# Step 2 — Train & evaluate models (temporal split)
# ======================================================================

def temporal_train_test_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sort by timestamp, then split into first (1-test_size) for train
    and the remaining test_size for test. Returns X_train, X_test,
    y_train, y_test.
    """
    # Sort the original DataFrame by timestamp and align X, y
    if "timestamp" in df.columns:
        sorted_idx = df["timestamp"].sort_values().index
    else:
        sorted_idx = df.index  # already in order

    X_sorted = X.loc[sorted_idx].reset_index(drop=True)
    y_sorted = y.loc[sorted_idx].reset_index(drop=True)

    split_point = int(len(X_sorted) * (1 - test_size))
    X_train, X_test = X_sorted.iloc[:split_point], X_sorted.iloc[split_point:]
    y_train, y_test = y_sorted.iloc[:split_point], y_sorted.iloc[split_point:]

    print(f"📊  Temporal split → Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def _evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Return RMSE, MAE, R² as a dict."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_models() -> Dict[str, Any]:
    """Return a dict of model_name → untrained estimator."""
    try:
        from xgboost import XGBRegressor
        xgb_model = XGBRegressor(**XGBOOST_PARAMS)
    except ImportError:
        print("⚠️  XGBoost not installed — skipping.")
        xgb_model = None

    models: Dict[str, Any] = {
        "RandomForest": RandomForestRegressor(
            n_estimators=N_ESTIMATORS_RF,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS_GBR,
            max_depth=8,
            learning_rate=LEARNING_RATE_GBR,
            random_state=RANDOM_STATE,
        ),
        "Ridge": Ridge(alpha=1.0),
    }
    if xgb_model is not None:
        models["XGBoost"] = xgb_model

    return models


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[str, Any, Dict[str, Dict[str, float]]]:
    """
    Train each model, evaluate, print comparison table, and return
    (best_name, best_model, all_results).
    """
    models = build_models()
    results: Dict[str, Dict[str, float]] = {}
    trained: Dict[str, Any] = {}

    for name, model in models.items():
        print(f"\n🔧  Training {name} …")
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = _evaluate(y_test, preds)
            results[name] = metrics
            trained[name] = model
            print(f"    RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")
        except Exception as exc:
            print(f"    ❌  {name} failed: {exc}")

    # ── Pretty comparison table ───────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Model':<22} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<22} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['r2']:>10.4f}")
    print("=" * 60)

    # ── Select best model by lowest RMSE ──────────────────────────────
    best_name = min(results, key=lambda n: results[n]["rmse"])
    print(f"\n🏆  Best model: {best_name} (RMSE={results[best_name]['rmse']:.4f})")
    return best_name, trained[best_name], results


# ======================================================================
# Step 3 — SHAP analysis
# ======================================================================

def compute_feature_importance(
    model: Any,
    model_name: str,
    X_test: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute SHAP-based feature importance.  Falls back to
    model.feature_importances_ (tree models) or coefficient magnitudes
    (linear models) when SHAP fails.
    """
    feature_names = list(X_test.columns)
    importance: Dict[str, float] = {}

    try:
        import shap

        print("🔍  Computing SHAP values …")
        if model_name in ("RandomForest", "GradientBoosting", "XGBoost"):
            explainer = shap.TreeExplainer(model)
        else:
            # Ridge / linear — use generic Explainer with a sample
            background = X_test.sample(min(100, len(X_test)), random_state=RANDOM_STATE)
            explainer = shap.Explainer(model, background)

        shap_values = explainer.shap_values(X_test.iloc[:200])  # cap for speed
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = {
            feat: float(val)
            for feat, val in zip(feature_names, mean_abs)
        }
        print("✅  SHAP importance computed successfully.")
    except Exception as exc:
        print(f"⚠️  SHAP failed ({exc}), using fallback …")
        if hasattr(model, "feature_importances_"):
            importance = {
                feat: float(val)
                for feat, val in zip(feature_names, model.feature_importances_)
            }
            print("✅  Used model.feature_importances_ as fallback.")
        elif hasattr(model, "coef_"):
            importance = {
                feat: float(abs(val))
                for feat, val in zip(feature_names, model.coef_)
            }
            print("✅  Used |model.coef_| as fallback.")
        else:
            # Last resort — uniform
            importance = {feat: 1.0 / len(feature_names) for feat in feature_names}
            print("⚠️  No importance method available; using uniform weights.")

    # Sort descending
    importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))

    top_5 = list(importance.items())[:5]
    print("   Top-5 features:")
    for feat, val in top_5:
        print(f"      {feat}: {val:.4f}")

    return importance


# ======================================================================
# Step 4 — Save model to Hopsworks Model Registry
# ======================================================================

def save_model_to_registry(
    model: Any,
    best_name: str,
    metrics: Dict[str, float],
    importance: Dict[str, float],
    feature_columns: List[str],
) -> None:
    """
    Persist the model + metadata locally, then upload to the Hopsworks
    Model Registry.
    """
    import hopsworks

    local_dir = "aqi_model"

    # Clean up previous run
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # ── Serialize artefacts ───────────────────────────────────────────
    print("💾  Saving model artifacts locally …")
    joblib.dump(model, os.path.join(local_dir, "model.pkl"))

    with open(os.path.join(local_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(local_dir, "feature_importance.json"), "w") as f:
        json.dump(importance, f, indent=2)

    with open(os.path.join(local_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    print(f"   Saved to ./{local_dir}/")

    # ── Upload to Hopsworks ───────────────────────────────────────────
    print("🚀  Uploading to Hopsworks Model Registry …")
    project = hopsworks.login()
    mr = project.get_model_registry()

    hw_model = mr.python.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description=f"Best AQI predictor model ({best_name})",
    )
    hw_model.save(local_dir)
    print(f"✅  Model '{MODEL_NAME}' uploaded successfully.")


# ======================================================================
# Main entry-point
# ======================================================================

def main() -> None:
    print("=" * 60)
    print("  Pearls AQI Predictor — Training Pipeline")
    print("=" * 60)

    # 1. Fetch data
    df = fetch_training_data()

    # 2. Prepare features
    X, y, feature_columns = prepare_features(df)

    # Handle NaN / Inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    valid_mask = y.notna()
    X, y, df = X[valid_mask], y[valid_mask], df[valid_mask]

    # 3. Temporal split
    X_train, X_test, y_train, y_test = temporal_train_test_split(df, X, y)

    # 4. Train & evaluate
    best_name, best_model, all_results = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # 5. SHAP importance
    importance = compute_feature_importance(best_model, best_name, X_test)

    # 6. Upload to Hopsworks
    best_metrics = all_results[best_name]
    save_model_to_registry(best_model, best_name, best_metrics, importance, feature_columns)

    print("\n🎉  Training pipeline finished successfully!")


if __name__ == "__main__":
    main()
