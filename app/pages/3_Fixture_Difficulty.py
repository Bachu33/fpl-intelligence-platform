import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
            "opponent": teams.get(fixture["team_a"], "Unknown") + " (H)",
            "gameweek": fixture["event"],
            "fdr": fixture["team_h_difficulty"],
            "venue": "H"
        })
        rows.append({
            "team": teams.get(fixture["team_a"], "Unknown"),
            "opponent": teams.get(fixture["team_h"], "Unknown") + " (A)",
            "gameweek": fixture["event"],
            "fdr": fixture["team_a_difficulty"],
            "venue": "A"
        })
    
    return pd.DataFrame(rows)

df = load_fixture_data()

if df.empty:
    st.warning("Could not load fixture data.")
    st.stop()

pivot_fdr = df.pivot_table(
    index="team",
    columns="gameweek",
    values="fdr",
    aggfunc="mean"
).round(1)

pivot_opponent = df.pivot_table(
    index="team",
    columns="gameweek",
    values="opponent",
    aggfunc="first"
)

pivot_fdr.columns = [f"GW{int(c)}" for c in pivot_fdr.columns]
pivot_opponent.columns = [f"GW{int(c)}" for c in pivot_opponent.columns]

pivot_fdr["avg_fdr"] = pivot_fdr.mean(axis=1)
pivot_fdr = pivot_fdr.sort_values("avg_fdr")
pivot_opponent = pivot_opponent.loc[pivot_fdr.index]

gw_cols = [c for c in pivot_fdr.columns if c != "avg_fdr"]
fdr_values = pivot_fdr[gw_cols]
opponent_values = pivot_opponent[gw_cols]

import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
    z=fdr_values.values,
    x=gw_cols,
    y=fdr_values.index.tolist(),
    text=opponent_values.values,
    texttemplate="%{text}",
    textfont={"size": 11, "color": "black"},
    colorscale=[[0, "#00ff87"], [0.25, "#00ff87"], [0.5, "#ebff00"], [0.75, "#ff7700"], [1.0, "#ff4c4c"]],
    zmin=1,
    zmax=5,
    showscale=True,
    colorbar=dict(
        title="FDR",
        tickvals=[1, 2, 3, 4, 5],
        ticktext=["1 - Very Easy", "2 - Easy", "3 - Medium", "4 - Hard", "5 - Very Hard"]
    )
))

fig.update_layout(
    title="Fixture Difficulty Rating — Next 6 Gameweeks",
    height=700,
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3"),
    xaxis=dict(side="top")
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("**Colour Key:**")
col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown('<div style="background-color:#00ff87;padding:8px;border-radius:4px;text-align:center;color:black"><b>1 — Very Easy</b></div>', unsafe_allow_html=True)
col2.markdown('<div style="background-color:#80ff44;padding:8px;border-radius:4px;text-align:center;color:black"><b>2 — Easy</b></div>', unsafe_allow_html=True)
col3.markdown('<div style="background-color:#ebff00;padding:8px;border-radius:4px;text-align:center;color:black"><b>3 — Medium</b></div>', unsafe_allow_html=True)
col4.markdown('<div style="background-color:#ff7700;padding:8px;border-radius:4px;text-align:center;color:white"><b>4 — Hard</b></div>', unsafe_allow_html=True)
col5.markdown('<div style="background-color:#ff4c4c;padding:8px;border-radius:4px;text-align:center;color:white"><b>5 — Very Hard</b></div>', unsafe_allow_html=True)

st.caption("Teams sorted by average FDR — easiest fixtures at the top. (H) = Home, (A) = Away.")