import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_supabase_client

st.set_page_config(page_title="Fixture Difficulty", page_icon="📅", layout="wide")

from utils import apply_custom_css
apply_custom_css()

st.title("📅 Fixture Difficulty Rating")
st.markdown("FDR heatmap showing how hard each team's upcoming fixtures are. Green = easy, Red = hard.")
st.markdown("---")

@st.cache_data(ttl=3600)
def load_fixture_data():
    headers = {"User-Agent": "fpl-intelligence-platform/1.0"}
    
    bootstrap = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/",
        headers=headers
    ).json()
    
    fixtures = requests.get(
        "https://fantasy.premierleague.com/api/fixtures/",
        headers=headers
    ).json()
    
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    
    current_gw = 1
    for event in bootstrap["events"]:
        if event["is_current"]:
            current_gw = event["id"]
            break
    
    upcoming = [f for f in fixtures if f["event"] and f["event"] >= current_gw and f["event"] < current_gw + 6]
    
    rows = []
    for fixture in upcoming:
        rows.append({
            "team": teams.get(fixture["team_h"], "Unknown"),
            "opponent": teams.get(fixture["team_a"], "Unknown"),
            "gameweek": fixture["event"],
            "fdr": fixture["team_h_difficulty"],
            "venue": "H"
        })
        rows.append({
            "team": teams.get(fixture["team_a"], "Unknown"),
            "opponent": teams.get(fixture["team_h"], "Unknown"),
            "gameweek": fixture["event"],
            "fdr": fixture["team_a_difficulty"],
            "venue": "A"
        })
    
    return pd.DataFrame(rows)

df = load_fixture_data()

if df.empty:
    st.warning("Could not load fixture data.")
    st.stop()

pivot = df.pivot_table(
    index="team",
    columns="gameweek",
    values="fdr",
    aggfunc="mean"
).round(1)

pivot.columns = [f"GW{int(c)}" for c in pivot.columns]

pivot["avg_fdr"] = pivot.mean(axis=1)
pivot = pivot.sort_values("avg_fdr")

fig = px.imshow(
    pivot.drop(columns=["avg_fdr"]),
    color_continuous_scale=["#00ff87", "#ebff00", "#ff4c4c"],
    range_color=[1, 5],
    labels={"color": "FDR"},
    title="Fixture Difficulty Rating — Next 6 Gameweeks (1=Easy, 5=Hard)",
    aspect="auto"
)

fig.update_layout(
    height=700,
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3")
)

st.plotly_chart(fig, use_container_width=True)

st.caption("Teams sorted by average FDR — easiest fixtures at the top.")