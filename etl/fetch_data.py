import requests
import json
import os
import time
from datetime import datetime

HEADERS = {"User-Agent": "fpl-intelligence-platform/1.0"}
ELEMENT_SUMMARY_DIR = "data/raw/element_summaries"

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

def fetch_element_summary(session, player_id):
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    response = session.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"WARNING: element-summary fetch failed for player {player_id} with status {response.status_code}")
        return None

    data = response.json()

    with open(f"{ELEMENT_SUMMARY_DIR}/{player_id}.json", "w") as f:
        json.dump(data, f)

    return data

def fetch_player_histories(bootstrap_data):
    os.makedirs(ELEMENT_SUMMARY_DIR, exist_ok=True)

    players = bootstrap_data["elements"]
    session = requests.Session()
    successful = 0

    print(f"Fetching historical summaries for {len(players)} players...")

    for idx, player in enumerate(players, start=1):
        player_id = player["id"]
        data = fetch_element_summary(session, player_id)
        if data is not None:
            successful += 1

        if idx % 50 == 0:
            print(f"  Fetched {idx}/{len(players)} player summaries")

        time.sleep(0.05)

    print(f"[{datetime.now()}] Player summaries saved. Successful: {successful}/{len(players)}")

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
    fetch_player_histories(bootstrap_data)
    
    print("ETL fetch complete.")

