import pandas as pd
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def load_raw_data():
    with open("data/raw/bootstrap.json", "r") as f:
        bootstrap = json.load(f)
    
    elements = bootstrap["elements"]
    teams = bootstrap["teams"]
    events = bootstrap["events"]
    
    print(f"Loaded {len(elements)} players, {len(teams)} teams")
    return elements, teams, events

def get_current_gw(events):
    for event in events:
        if event["is_current"]:
            return event["id"]
    for event in reversed(events):
        if event["is_finished"]:
            return event["id"]
    return 1

def process_players(elements, teams, current_gw):
    teams_dict = {team["id"]: team["name"] for team in teams}
    
    position_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    
    records = []
    
    for player in elements:
        record = {
            "player_id": player["id"],
            "player_name": player["web_name"],
            "team": teams_dict.get(player["team"], "Unknown"),
            "position": position_map.get(player["element_type"], "Unknown"),
            "now_cost": player["now_cost"],
            "selected_by_percent": float(player["selected_by_percent"]),
            "total_points": player["total_points"],
            "form": float(player["form"]),
            "ict_index": float(player["ict_index"]),
            "influence": float(player["influence"]),
            "creativity": float(player["creativity"]),
            "threat": float(player["threat"]),
            "transfers_in": player["transfers_in_event"],
            "transfers_out": player["transfers_out_event"],
            "minutes": player["minutes"],
            "goals_scored": player["goals_scored"],
            "assists": player["assists"],
            "clean_sheets": player["clean_sheets"],
            "goals_conceded": player["goals_conceded"],
            "yellow_cards": player["yellow_cards"],
            "red_cards": player["red_cards"],
            "saves": player["saves"],
            "bonus": player["bonus"],
            "bps": player["bps"],
            "points_per_game": float(player["points_per_game"]),
            "gameweek": current_gw,
            "season": "2024-25"
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/players.parquet", index=False)
    
    print(f"[{datetime.now()}] Processed {len(df)} player records")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3))
    
    return df

def push_to_supabase(df):
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("WARNING: SUPABASE_URL or SUPABASE_KEY not found. Skipping Supabase push.")
        return

    try:
        from supabase import create_client
    except ModuleNotFoundError:
        print("WARNING: 'supabase' package not installed. Skipping Supabase push.")
        return

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client connected.")

    records = df.to_dict(orient="records")
    print(f"Pushing {len(records)} records to Supabase...")

    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        supabase.table("player_gameweek_stats").upsert(batch).execute()
        print(f"  Uploaded records {i + 1} to {min(i + batch_size, len(records))}")

    print(f"[{datetime.now()}] All records pushed successfully.")

if __name__ == "__main__":
    print("Starting data processing...")
    elements, teams, events = load_raw_data()
    current_gw = get_current_gw(events)
    print(f"Processing for GW{current_gw}")
    df = process_players(elements, teams, current_gw)
    push_to_supabase(df)
    print("Processing complete.")