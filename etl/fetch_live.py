import requests
import os
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]  # service_role key
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_current_gameweek():
    r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    r.raise_for_status()
    events = r.json()["events"]
    for e in events:
        if e["is_current"]:
            return e["id"], not e["finished"]
    return None, False

def fetch_live_stats(event_id):
    r = requests.get(f"https://fantasy.premierleague.com/api/event/{event_id}/live/")
    r.raise_for_status()
    return r.json()["elements"]

def push_live_stats(event_id, elements):
    rows = []
    for el in elements:
        stats = el["stats"]
        rows.append({
            "gameweek": event_id,
            "player_id": el["id"],
            "minutes": stats["minutes"],
            "goals_scored": stats["goals_scored"],
            "assists": stats["assists"],
            "bonus": stats["bonus"],
            "bps": stats["bps"],
            "total_points": stats["total_points"],
        })
    # batch upsert, 500 rows at a time
    for i in range(0, len(rows), 500):
        batch = rows[i:i+500]
        supabase.table("player_live_stats").upsert(
            batch, on_conflict="gameweek,player_id"
        ).execute()
    print(f"Pushed {len(rows)} live records for GW{event_id}")

if __name__ == "__main__":
    gw_id, is_live = get_current_gameweek()
    if gw_id is None or not is_live:
        print("No live gameweek right now. Exiting.")
        exit(0)
    elements = fetch_live_stats(gw_id)
    push_live_stats(gw_id, elements)