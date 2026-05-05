import os
import argparse
from datetime import datetime

import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

PREDICTION_FEATURES_PATH = "data/processed/prediction_features.parquet"
LOCAL_PREDICTIONS_PATH = "data/processed/latest_predictions.csv"
CATEGORICAL_COLUMNS = ["position", "team", "opponent_team_id"]


def load_artifacts():
    required = [
        "model/xgb_model.joblib",
        "model/encoders.joblib",
        "model/feature_columns.joblib",
    ]

    for path in required:
        if not os.path.exists(path):
            print(f"FATAL: {path} not found. Run model/train.py first.")
            raise SystemExit(1)

    model = joblib.load("model/xgb_model.joblib")
    encoders = joblib.load("model/encoders.joblib")
    feature_columns = joblib.load("model/feature_columns.joblib")

    print("Model, encoders, and feature list loaded.")
    return model, encoders, feature_columns


def load_prediction_features():
    if not os.path.exists(PREDICTION_FEATURES_PATH):
        print(f"FATAL: {PREDICTION_FEATURES_PATH} not found. Run etl/build_gw_dataset.py first.")
        raise SystemExit(1)

    df = pd.read_parquet(PREDICTION_FEATURES_PATH)
    print(f"Loaded {len(df)} players for prediction")
    return df


def encode_known_categories(df, encoders):
    df = df.copy()

    for col in CATEGORICAL_COLUMNS:
        encoder = encoders[col]
        values = df[col].fillna("Unknown").astype(str)
        known = set(encoder.classes_)
        df = df[values.isin(known)].copy()
        df[f"{col}_encoded"] = encoder.transform(df[col].fillna("Unknown").astype(str))

    return df


def prepare_features(df, encoders, feature_columns):
    df = encode_known_categories(df, encoders)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df[df["fixture_count"] > 0].copy()

    print(f"Features prepared for {len(df)} players with upcoming fixtures")
    return df


def run_predictions(df, model, feature_columns):
    predictions = model.predict(df[feature_columns])

    df = df.copy()
    df["predicted_points"] = predictions.round(2)
    df = df.sort_values("predicted_points", ascending=False)

    print("\nTop 10 Predicted Players This GW:")
    print(df[["player_name", "team", "position", "predicted_points"]].head(10).to_string(index=False))

    return df


def push_predictions(df):
    if df.empty:
        print("FATAL: No predictions generated. Existing Supabase predictions were left untouched.")
        raise SystemExit(1)

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("FATAL: Supabase credentials not found. Local predictions were saved before upload.")
        raise SystemExit(1)

    try:
        from supabase import create_client
    except ModuleNotFoundError:
        print("FATAL: Python package 'supabase' is not installed. Run: python -m pip install -r requirements.txt")
        print("Local predictions were saved before upload.")
        raise SystemExit(1)

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase connected.")

    records = df[[
        "player_id",
        "player_name",
        "team",
        "position",
        "gameweek",
        "season",
        "predicted_points",
        "now_cost",
    ]].to_dict(orient="records")

    print(f"\nPushing {len(records)} predictions to Supabase...")

    supabase.table("predictions").delete().neq("id", 0).execute()

    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("predictions").insert(batch).execute()
        print(f"  Inserted records {i + 1} to {min(i + batch_size, len(records))}")

    print(f"[{datetime.now()}] Predictions pushed successfully.")


def save_local_predictions(df):
    os.makedirs(os.path.dirname(LOCAL_PREDICTIONS_PATH), exist_ok=True)
    df[[
        "player_id",
        "player_name",
        "team",
        "position",
        "gameweek",
        "season",
        "predicted_points",
        "now_cost",
    ]].to_csv(LOCAL_PREDICTIONS_PATH, index=False)
    print(f"\nSaved local predictions to {LOCAL_PREDICTIONS_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate FPL gameweek predictions.")
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Generate and save local predictions without uploading to Supabase.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 50)
    print("FPL Gameweek Predictions Generator")
    print("=" * 50)

    model, encoders, feature_columns = load_artifacts()
    data = load_prediction_features()
    data = prepare_features(data, encoders, feature_columns)
    predictions = run_predictions(data, model, feature_columns)
    save_local_predictions(predictions)

    if not args.skip_push:
        push_predictions(predictions)
    else:
        print("Skipped Supabase upload.")

    print("\nPrediction run complete.")
