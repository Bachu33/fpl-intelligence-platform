import streamlit as st
import pandas as pd
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_predictions, apply_custom_css, POSITION_COLORS, POSITION_ORDER, render_kicker, render_sidebar_nav

st.set_page_config(page_title="My Team", page_icon="👤", layout="wide")
apply_custom_css()
render_sidebar_nav("My Team")
render_kicker("My Team")

st.title("👤 My Team")
st.markdown("Enter your FPL team ID to get personalised recommendations based on your current squad.")
st.markdown("---")

st.info("**How to find your Team ID:** Go to the FPL website → Points → your URL will contain your team ID. Example: `https://fantasy.premierleague.com/entry/1234567/event/34` → your ID is `1234567`")

team_id = st.text_input("Enter your FPL Team ID", placeholder="e.g. 1234567")

HEADERS = {"User-Agent": "fpl-intelligence-platform/1.0"}

@st.cache_data(ttl=3600)
def fetch_my_team(team_id):
    gw_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/"
    
    bootstrap = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        headers=HEADERS
    ).json()
    
    current_gw = 1
    for event in bootstrap["events"]:
        if event["is_current"]:
            current_gw = event["id"]
            break
    
    picks_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{current_gw}/picks/"
    picks_response = requests.get(picks_url, headers=HEADERS)
    
    if picks_response.status_code != 200:
        return None, None, None
    
    picks_data = picks_response.json()
    
    players_dict = {p["id"]: p for p in bootstrap["elements"]}
    teams_dict = {t["id"]: t["name"] for t in bootstrap["teams"]}
    position_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    
    my_players = []
    for pick in picks_data["picks"]:
        player = players_dict.get(pick["element"], {})
        my_players.append({
            "player_id": pick["element"],
            "player_name": player.get("web_name", "Unknown"),
            "team": teams_dict.get(player.get("team"), "Unknown"),
            "position": position_map.get(player.get("element_type"), "Unknown"),
            "now_cost": player.get("now_cost", 0) / 10,
            "is_captain": pick["is_captain"],
            "is_vice_captain": pick["is_vice_captain"],
            "multiplier": pick["multiplier"],
            "form": float(player.get("form", 0)),
            "total_points": player.get("total_points", 0)
        })
    
    entry_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/"
    entry_response = requests.get(entry_url, headers=HEADERS)
    entry_data = entry_response.json() if entry_response.status_code == 200 else {}
    
    return pd.DataFrame(my_players), picks_data, entry_data

if team_id:
    with st.spinner("Fetching your team..."):
        my_team_df, picks_data, entry_data = fetch_my_team(team_id)
    
    if my_team_df is None:
        st.error("Could not fetch team. Check your Team ID and try again.")
        st.stop()
    
    team_name = entry_data.get("name", f"Team {team_id}")
    manager = f"{entry_data.get('player_first_name', '')} {entry_data.get('player_last_name', '')}".strip()
    overall_rank = entry_data.get("summary_overall_rank", "N/A")
    overall_points = entry_data.get("summary_overall_points", "N/A")
    
    st.subheader(f"🏆 {team_name}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Manager", manager if manager else "N/A")
    col2.metric("Overall Points", overall_points)
    col3.metric("Overall Rank", f"{overall_rank:,}" if isinstance(overall_rank, int) else overall_rank)
    
    st.markdown("---")
    
    starters = my_team_df[my_team_df["multiplier"] > 0].copy()
    bench = my_team_df[my_team_df["multiplier"] == 0].copy()
    
    st.subheader("Starting XI")
    
    for pos in POSITION_ORDER:
        pos_players = starters[starters["position"] == pos]
        if pos_players.empty:
            continue
        st.markdown(f"#### {pos}")
        for _, row in pos_players.iterrows():
            suffix = " 👑 Captain" if row["is_captain"] else " 🔰 Vice Captain" if row["is_vice_captain"] else ""
            st.markdown(f"- **{row['player_name']}** ({row['team']}) — £{row['now_cost']:.1f}m | Form: {row['form']}{suffix}")
    
    st.markdown("---")
    st.subheader("🪑 Bench")
    for _, row in bench.iterrows():
        st.markdown(f"- {row['player_name']} ({row['team']}) — {row['position']} | £{row['now_cost']:.1f}m")
    
    st.markdown("---")
    
    predictions_df = load_predictions()
    
    if not predictions_df.empty:
        st.subheader("💡 Transfer Recommendations")
        st.markdown("Players in your squad ranked by predicted points, with suggested upgrades from the same position and similar price range.")
        
        my_ids = set(my_team_df["player_id"].tolist())
        not_in_team = predictions_df[~predictions_df["player_id"].isin(my_ids)].copy()
        
        for pos in POSITION_ORDER:
            my_pos_players = starters[starters["position"] == pos].sort_values("total_points")
            
            if my_pos_players.empty:
                continue
            
            weakest = my_pos_players.iloc[0]
            
            upgrades = not_in_team[
                (not_in_team["position"] == pos) &
                (not_in_team["price"] <= weakest["now_cost"] + 1.5)
            ].sort_values("predicted_points", ascending=False).head(3)
            
            if upgrades.empty:
                continue
            
            st.markdown(f"#### {pos} — Consider replacing **{weakest['player_name']}**")
            
            upgrade_display = upgrades[["player_name", "team", "price", "predicted_points"]].copy()
            upgrade_display.columns = ["Player", "Team", "Price (£m)", "Predicted Pts"]
            upgrade_display["Price (£m)"] = upgrade_display["Price (£m)"].map("{:.1f}".format)
            upgrade_display["Predicted Pts"] = upgrade_display["Predicted Pts"].map("{:.2f}".format)
            upgrade_display = upgrade_display.reset_index(drop=True)
            upgrade_display.index += 1
            st.dataframe(upgrade_display, use_container_width=True)
