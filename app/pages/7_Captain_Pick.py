import os
import sys

import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    POSITION_COLORS,
    POSITION_ORDER,
    apply_custom_css,
    load_predictions,
    render_app_header,
    render_player_card,
)

st.set_page_config(page_title="Captain Pick", page_icon="👑", layout="wide")
apply_custom_css()

st.title("👑 Captain Recommendations")
st.markdown("---")

df = load_predictions()

if df.empty:
    st.warning("No predictions available yet.")
    st.stop()

render_app_header(
    "Captain Board",
    "Compare armband candidates with doubled points, fixture tags, and minutes-risk context.",
    badges=[f"GW {int(df['gameweek'].max())}", "2x scoring", "Risk aware"],
)

st.sidebar.header("Filters")
max_price = st.sidebar.slider("Max Price (£m)", 4.0, 15.0, 15.0, 0.5)
positions = st.sidebar.multiselect("Position", POSITION_ORDER, default=["MID", "FWD"])
include_minutes_risk = st.sidebar.checkbox("Include minutes-risk players", value=True)

filtered = df[(df["position"].isin(positions)) & (df["price"] <= max_price)].copy()
if not include_minutes_risk and "risk_label" in filtered.columns:
    filtered = filtered[filtered["risk_label"] != "Minutes risk"]

filtered = filtered.sort_values("predicted_points", ascending=False).head(10)

st.subheader("Captain Board")

cols = st.columns(2)
for idx, (_, row) in enumerate(filtered.head(4).iterrows(), start=1):
    with cols[(idx - 1) % 2]:
        render_player_card(row, rank=idx, captain=True)

st.markdown("---")

captain_table = filtered.copy()
captain_table["captain_points"] = captain_table["predicted_points"] * 2

display_cols = ["player_name", "team", "position", "price", "predicted_points", "captain_points"]
for optional in ["fixture_label", "opponent", "fdr", "risk_label"]:
    if optional in captain_table.columns:
        display_cols.append(optional)

display = captain_table[display_cols].copy()
display.columns = [
    "Player",
    "Team",
    "Position",
    "Price (£m)",
    "Predicted Pts",
    "Captain Pts",
    *[col.replace("_", " ").title() for col in display_cols[6:]],
]
for col in ["Price (£m)", "Predicted Pts", "Captain Pts"]:
    display[col] = display[col].map("{:.2f}".format if col != "Price (£m)" else "{:.1f}".format)
display = display.reset_index(drop=True)
display.index += 1
st.dataframe(display, use_container_width=True)

st.markdown("---")
st.subheader("Captain Points Comparison")

fig = px.bar(
    captain_table.sort_values("captain_points", ascending=True),
    x="captain_points",
    y="player_name",
    color="position",
    color_discrete_map=POSITION_COLORS,
    orientation="h",
    labels={"captain_points": "Captain Points", "player_name": "Player"},
    hover_data=["team", "price", "predicted_points"],
)

fig.update_layout(
    height=450,
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3"),
)

st.plotly_chart(fig, use_container_width=True)
