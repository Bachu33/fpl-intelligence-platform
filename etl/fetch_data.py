import requests
import json
import os
from datetime import datetime

HEADERS = {"User-Agent": "fpl-intelligence-platform/1.0"}

def fetch_bootstrap():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"ERROR: bootstrap fetch failed with status {response.status_code}")
        return None
    
    data = response.json()
    
    with open("data/raw/bootstrap.json", "w") as f:
        json.dump(data, f)
    
    print(f"[{datetime.now()}] Bootstrap data saved. Players found: {len(data['elements'])}")
    return data

def fetch_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"ERROR: fixtures fetch failed with status {response.status_code}")
        return None
    
    data = response.json()
    
    with open("data/raw/fixtures.json", "w") as f:
        json.dump(data, f)
    
    print(f"[{datetime.now()}] Fixtures saved. Total fixtures: {len(data)}")
    return data

def fetch_gw_live(gw):
    url = f"https://fantasy.premierleague.com/api/event/{gw}/live/"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"ERROR: GW{gw} live fetch failed with status {response.status_code}")
        return None
    
    data = response.json()
    
    with open(f"data/raw/gw_{gw}_live.json", "w") as f:
        json.dump(data, f)
    
    print(f"[{datetime.now()}] GW{gw} live data saved. Entries: {len(data['elements'])}")
    return data

def get_current_gameweek(bootstrap_data):
    events = bootstrap_data["events"]
    for event in events:
        if event["is_current"]:
            print(f"Current gameweek: {event['id']}")
            return event["id"]
    
    print("WARNING: No current gameweek found. Defaulting to most recent finished GW.")
    for event in reversed(events):
        if event["is_finished"]:
            return event["id"]
    
    return 1

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("Starting FPL data fetch...")
    
    bootstrap_data = fetch_bootstrap()
    if bootstrap_data is None:
        print("FATAL: Could not fetch bootstrap. Exiting.")
        exit(1)
    
    fetch_fixtures()
    
    current_gw = get_current_gameweek(bootstrap_data)
    fetch_gw_live(current_gw)
    
    print("ETL fetch complete.")

