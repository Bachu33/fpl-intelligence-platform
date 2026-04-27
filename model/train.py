import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

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

TARGET_COLUMN = "points_per_game"

def load_data():
    path = "data/processed/players.parquet"
    
    if not os.path.exists(path):
        print("ERROR: players.parquet not found. Run process_data.py first.")
        exit(1)
    
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} records from parquet")
    print(f"Columns: {df.columns.tolist()}")
    return df

def preprocess(df):
    df = df.copy()
    
    for col in ["form", "ict_index", "influence", "creativity", 
                "threat", "selected_by_percent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    position_encoder = LabelEncoder()
    df["position_encoded"] = position_encoder.fit_transform(df["position"])
    
    team_encoder = LabelEncoder()
    df["team_encoded"] = team_encoder.fit_transform(df["team"])
    
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    
    df = df[df["minutes"] > 0]
    
    print(f"After preprocessing: {len(df)} records remain")
    print(f"Position encoding: {dict(zip(position_encoder.classes_, position_encoder.transform(position_encoder.classes_)))}")
    
    return df, position_encoder, team_encoder

def train_model(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")
    
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  MAE:  {mae:.3f} points")
    print(f"  R²:   {r2:.3f}")
    
    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"\nTop 5 Features:")
    print(feature_importance.head(5).to_string(index=False))
    
    return model

def save_artifacts(model, position_encoder, team_encoder):
    os.makedirs("model", exist_ok=True)
    
    joblib.dump(model, "model/xgb_model.joblib")
    joblib.dump(position_encoder, "model/position_encoder.joblib")
    joblib.dump(team_encoder, "model/team_encoder.joblib")
    
    print(f"\n[{datetime.now()}] Saved:")
    print(f"  model/xgb_model.joblib")
    print(f"  model/position_encoder.joblib")
    print(f"  model/team_encoder.joblib")

if __name__ == "__main__":
    print("=" * 50)
    print("FPL XGBoost Model Training")
    print("=" * 50)
    
    df = load_data()
    df, position_encoder, team_encoder = preprocess(df)
    model = train_model(df)
    save_artifacts(model, position_encoder, team_encoder)
    
    print("\nTraining complete.")

