import pandas as pd
import joblib
import os
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("FATAL: Supabase credentials not found.")
    exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Supabase connected.")

FEATURE_COLUMNS = [
    "now_cost",
    "selected_by_percent",
    "form",
    "ict_index",
    "influence",
    "creativity",
    "threat",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "transfers_in",
    "transfers_out",
    "position_encoded",
    "team_encoded"
]

def load_artifacts():
    required = [
        "model/xgb_model.joblib",
        "model/position_encoder.joblib",
        "model/team_encoder.joblib"
    ]
    
    for path in required:
        if not os.path.exists(path):
            print(f"FATAL: {path} not found. Run train.py first.")
            exit(1)
    
    model = joblib.load("model/xgb_model.joblib")
    position_encoder = joblib.load("model/position_encoder.joblib")
    team_encoder = joblib.load("model/team_encoder.joblib")
    
    print("Model and encoders loaded.")
    return model, position_encoder, team_encoder

def load_current_data():
    path = "data/processed/players.parquet"
    
    if not os.path.exists(path):
        print("FATAL: players.parquet not found. Run process_data.py first.")
        exit(1)
    
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} players for prediction")
    return df

def prepare_features(df, position_encoder, team_encoder):
    df = df.copy()
    
    for col in ["form", "ict_index", "influence", "creativity",
                "threat", "selected_by_percent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    known_positions = list(position_encoder.classes_)
    df = df[df["position"].isin(known_positions)]
    df["position_encoded"] = position_encoder.transform(df["position"])
    
    known_teams = list(team_encoder.classes_)
    df = df[df["team"].isin(known_teams)]
    df["team_encoded"] = team_encoder.transform(df["team"])
    
    df = df.dropna(subset=FEATURE_COLUMNS)

    current_gw = df["gameweek"].max()
    min_minutes = current_gw * 45
    df = df[df["minutes"] >= min_minutes]
    
    print(f"Features prepared for {len(df)} players")
    return df

def run_predictions(df, model):
    X = df[FEATURE_COLUMNS]
    
    predictions = model.predict(X)
    
    df = df.copy()
    df["predicted_points"] = predictions
    df["predicted_points"] = df["predicted_points"].round(2)
    
    df = df.sort_values("predicted_points", ascending=False)
    
    print(f"\nTop 10 Predicted Players This GW:")
    print(df[["player_name", "team", "position", "predicted_points"]].head(10).to_string(index=False))
    
    return df

def push_predictions(df):
    records = df[["player_id", "player_name", "team", "position", 
                  "gameweek", "season", "predicted_points"]].to_dict(orient="records")
    
    print(f"\nPushing {len(records)} predictions to Supabase...")
    
    supabase.table("predictions").delete().neq("id", 0).execute()
    
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("predictions").insert(batch).execute()
        print(f"  Inserted records {i+1} to {min(i+batch_size, len(records))}")
    
    print(f"[{datetime.now()}] Predictions pushed successfully.")

if __name__ == "__main__":
    print("=" * 50)
    print("FPL Predictions Generator")
    print("=" * 50)
    
    model, position_encoder, team_encoder = load_artifacts()
    df = load_current_data()
    df = prepare_features(df, position_encoder, team_encoder)
    df = run_predictions(df, model)
    push_predictions(df)
    
    print("\nPrediction run complete.")

