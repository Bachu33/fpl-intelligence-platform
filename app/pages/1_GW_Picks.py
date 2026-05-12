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

st.set_page_config(page_title="GW Picks — FPL Intelligence", page_icon="🎯", layout="wide")
apply_custom_css()
render_sidebar_nav("GW Picks")

df = load_predictions()

if df.empty:
    st.warning("No predictions available yet. The pipeline may not have run this gameweek.")
    st.stop()

current_gw = int(df["gameweek"].max())
render_kicker("GW Picks")
render_app_header(
    f"Predicted Points · GW {current_gw}",
    "Filter the model board by position, budget, fixture type, and minutes risk.",
    badges=["Position filters", "Budget", "Fixture tags"],
)

with st.sidebar:
    st.markdown("### Filters")
    pos = st.radio("Position", ["ALL"] + POSITION_ORDER, horizontal=True)
    max_price = st.slider("Max price (£m)", min_value=4.0, max_value=15.0, value=15.0, step=0.5)
    top_n = st.slider("Show top", min_value=10, max_value=200, value=50, step=10)
    hide_minutes_risk = st.checkbox("Hide minutes-risk players", value=False)
    fixture_mode = st.selectbox("Fixture type", ["All", "Double GW", "Single GW"])

filtered = df[df["price"] <= max_price].copy()
if pos != "ALL":
    filtered = filtered[filtered["position"] == pos]
if hide_minutes_risk and "risk_label" in filtered.columns:
    filtered = filtered[filtered["risk_label"] != "Minutes risk"]
if fixture_mode == "Double GW" and "fixture_count" in filtered.columns:
    filtered = filtered[filtered["fixture_count"] >= 2]
elif fixture_mode == "Single GW" and "fixture_count" in filtered.columns:
    filtered = filtered[filtered["fixture_count"] < 2]

filtered = filtered.sort_values("predicted_points", ascending=False).head(top_n)

st.caption(f"Showing {len(filtered)} players")
card_cols = st.columns(3)
for idx, (_, row) in enumerate(filtered.head(6).iterrows(), start=1):
    with card_cols[(idx - 1) % 3]:
        render_player_card(row, rank=idx)

display_cols = ["player_name", "team", "position", "price", "predicted_points"]
for optional in ["fixture_label", "opponent", "fdr", "risk_label", "value_score"]:
    if optional in filtered.columns:
        display_cols.append(optional)

display = filtered[display_cols].copy()
display.columns = [
    "Player",
    "Team",
    "Pos",
    "Price (£m)",
    "Predicted Pts",
    *[col.replace("_", " ").title() for col in display_cols[5:]],
]
display.insert(0, "#", range(1, len(display) + 1))

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
    height=620,
    column_config=prediction_column_config(),
)

st.markdown("### Predicted Points Distribution")
fig = px.bar(
    filtered.sort_values("predicted_points", ascending=True),
    x="predicted_points",
    y="player_name",
    color="position",
    color_discrete_map=POSITION_COLORS,
    orientation="h",
    labels={"predicted_points": "Predicted Points", "player_name": "Player"},
)
fig.update_layout(
    height=max(420, min(900, len(filtered) * 24)),
    paper_bgcolor="#161b22",
    plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3"),
)
fig.update_xaxes(gridcolor="#30363d")
fig.update_yaxes(gridcolor="#30363d")
st.plotly_chart(fig, use_container_width=True)
