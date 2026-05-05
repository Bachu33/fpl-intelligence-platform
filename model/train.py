import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

TRAINING_PATH = "data/processed/player_gameweek_features.parquet"
TARGET_COLUMN = "target_points"

CATEGORICAL_COLUMNS = ["position", "team", "opponent_team_id"]
BASE_FEATURE_COLUMNS = [
    "fdr",
    "fixture_count",
    "was_home",
    "games_played_before",
]

def load_data():
    if not os.path.exists(TRAINING_PATH):
        print(f"ERROR: {TRAINING_PATH} not found. Run etl/build_gw_dataset.py first.")
        raise SystemExit(1)

    df = pd.read_parquet(TRAINING_PATH)
    print(f"Loaded {len(df)} player-gameweek rows from {TRAINING_PATH}")
    return df


def get_feature_columns(df):
    lag_columns = [
        col for col in df.columns
        if col.endswith("_lag1") or col.endswith("_roll3") or col.endswith("_roll5")
    ]
    encoded_columns = [f"{col}_encoded" for col in CATEGORICAL_COLUMNS]
    return BASE_FEATURE_COLUMNS + lag_columns + encoded_columns


def preprocess(df):
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        values = df[col].fillna("Unknown").astype(str)
        df[f"{col}_encoded"] = encoder.fit_transform(values)
        encoders[col] = encoder

    feature_columns = get_feature_columns(df)

    for col in feature_columns + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feature_columns + [TARGET_COLUMN, "round"])
    df = df[df["fixture_count"] > 0]

    print(f"After preprocessing: {len(df)} rows remain")
    print(f"Gameweek range: GW{int(df['round'].min())}-GW{int(df['round'].max())}")
    print(f"Features: {len(feature_columns)}")

    return df, encoders, feature_columns


def make_model():
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )


def rolling_backtest(df, feature_columns, min_train_gw=5):
    gameweeks = sorted(int(gw) for gw in df["round"].dropna().unique())
    results = []
    predictions = []

    for gw in gameweeks:
        train_df = df[df["round"] < gw]
        test_df = df[df["round"] == gw]

        if gw < min_train_gw or train_df.empty or test_df.empty:
            continue

        model = make_model()
        model.fit(train_df[feature_columns], train_df[TARGET_COLUMN])
        y_pred = model.predict(test_df[feature_columns])

        mae = mean_absolute_error(test_df[TARGET_COLUMN], y_pred)
        r2 = r2_score(test_df[TARGET_COLUMN], y_pred) if len(test_df) > 1 else np.nan

        results.append({
            "gameweek": gw,
            "rows": len(test_df),
            "mae": mae,
            "r2": r2,
        })

        fold_predictions = test_df[["player_id", "player_name", "round", TARGET_COLUMN]].copy()
        fold_predictions["predicted_points"] = y_pred
        predictions.append(fold_predictions)

    if not results:
        print("WARNING: Not enough history for rolling backtest.")
        return pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(predictions, ignore_index=True)

    print("\nRolling Backtest:")
    print(results_df.to_string(index=False, formatters={
        "mae": "{:.3f}".format,
        "r2": "{:.3f}".format,
    }))
    print(f"\nOverall rolling MAE: {results_df['mae'].mean():.3f} points")
    print(f"Overall rolling R2:  {results_df['r2'].mean():.3f}")

    return results_df, predictions_df


def train_final_model(df, feature_columns):
    model = make_model()
    model.fit(df[feature_columns], df[TARGET_COLUMN])
    return model


def save_artifacts(model, encoders, feature_columns, backtest):
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/xgb_model.joblib")
    joblib.dump(encoders, "model/encoders.joblib")
    joblib.dump(feature_columns, "model/feature_columns.joblib")

    if not backtest.empty:
        backtest.to_csv("model/backtest_metrics.csv", index=False)

    print(f"\n[{datetime.now()}] Saved:")
    print("  model/xgb_model.joblib")
    print("  model/encoders.joblib")
    print("  model/feature_columns.joblib")
    if not backtest.empty:
        print("  model/backtest_metrics.csv")


if __name__ == "__main__":
    print("=" * 50)
    print("FPL Gameweek Forecast Model Training")
    print("=" * 50)

    data = load_data()
    data, encoders, feature_columns = preprocess(data)
    backtest, _ = rolling_backtest(data, feature_columns)
    final_model = train_final_model(data, feature_columns)
    save_artifacts(final_model, encoders, feature_columns, backtest)

    print("\nTraining complete.")
