import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_player_stats, POSITION_ORDER

st.set_page_config(page_title="Form Heatmap", page_icon="🔥", layout="wide")

st.title("🔥 Form Heatmap")
st.markdown("Players ranked by current form score (rolling average of recent gameweek points).")
st.markdown("---")

df = load_player_stats()

if df.empty:
    st.warning("No player data available yet.")
    st.stop()

st.sidebar.header("Filters")

position = st.sidebar.selectbox(
    "Position",
    options=["All"] + POSITION_ORDER
)

top_n = st.sidebar.slider("Show Top N Players", 10, 40, 20, 5)

if position != "All":
    df = df[df["position"] == position]

df = df[df["minutes"] > 0]
df = df.sort_values("form", ascending=False).head(top_n)

st.subheader(f"Top {top_n} Players by Form — {position}")

fig = px.bar(
    df.sort_values("form", ascending=True),
    x="form",
    y="player_name",
    color="form",
    color_continuous_scale="RdYlGn",
    orientation="h",
    labels={"form": "Form Score", "player_name": "Player"},
    hover_data=["team", "position", "points_per_game", "price"]
)

fig.update_layout(height=max(400, top_n * 25), coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Form vs ICT Index")

fig2 = px.scatter(
    df,
    x="ict_index",
    y="form",
    color="position",
    size="price",
    hover_name="player_name",
    hover_data=["team", "points_per_game"],
    labels={"ict_index": "ICT Index", "form": "Form Score"},
    title="Form vs ICT Index (bubble size = price)"
)

st.plotly_chart(fig2, use_container_width=True)