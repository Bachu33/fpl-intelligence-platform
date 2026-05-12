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
    prediction_column_config,
    render_app_header,
    render_kicker,
    render_player_card,
    render_sidebar_nav,
)

st.set_page_config(page_title="Captain Pick — FPL Intelligence", page_icon="👑", layout="wide")
apply_custom_css()
render_sidebar_nav("Captain Pick")

df = load_predictions()

if df.empty:
    st.warning("No predictions available yet.")
    st.stop()

current_gw = int(df["gameweek"].max())
render_kicker("Captain")
render_app_header(
    f"Captain Picks · GW {current_gw}",
    "Doubled predicted points ranking with fixture tags and minutes-risk context.",
    badges=["2x scoring", "Risk aware", "Fixture context"],
)

with st.sidebar:
    st.markdown("### Filters")
    max_price = st.slider("Max price (£m)", 4.0, 15.0, 15.0, 0.5)
    positions = st.multiselect("Position", POSITION_ORDER, default=["MID", "FWD"])
    include_minutes_risk = st.checkbox("Include minutes-risk players", value=True)

filtered = df[(df["position"].isin(positions)) & (df["price"] <= max_price)].copy()
if not include_minutes_risk and "risk_label" in filtered.columns:
    filtered = filtered[filtered["risk_label"] != "Minutes risk"]
filtered = filtered.sort_values("predicted_points", ascending=False).head(10)

cols = st.columns(2)
for idx, (_, row) in enumerate(filtered.head(4).iterrows(), start=1):
    with cols[(idx - 1) % 2]:
        render_player_card(row, rank=idx, captain=True)

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
    "Pos",
    "Price (£m)",
    "Predicted Pts",
    "Captain Pts",
    *[col.replace("_", " ").title() for col in display_cols[6:]],
]
display.insert(0, "Rank", range(1, len(display) + 1))

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    column_config=prediction_column_config(max_points=15),
)

st.markdown("### Captain Points Comparison")
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
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3"),
)
fig.update_xaxes(gridcolor="#30363d")
fig.update_yaxes(gridcolor="#30363d")
st.plotly_chart(fig, use_container_width=True)
