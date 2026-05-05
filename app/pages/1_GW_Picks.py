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
    render_player_card,
)

st.set_page_config(page_title="GW Picks", page_icon="🎯", layout="wide")
apply_custom_css()

st.title("🎯 GW Picks")
st.markdown("---")

df = load_predictions()

if df.empty:
    st.warning("No predictions available yet. The pipeline may not have run this gameweek.")
    st.stop()

st.sidebar.header("Filters")

positions = st.sidebar.multiselect("Position", options=POSITION_ORDER, default=POSITION_ORDER)
max_price = st.sidebar.slider("Max Price (£m)", min_value=4.0, max_value=15.0, value=15.0, step=0.5)
top_n = st.sidebar.slider("Show Top N Players", min_value=5, max_value=50, value=20, step=5)
hide_minutes_risk = st.sidebar.checkbox("Hide minutes-risk players", value=False)
fixture_mode = st.sidebar.selectbox("Fixture Type", ["All", "Double GW", "Single GW"])

filtered = df[(df["position"].isin(positions)) & (df["price"] <= max_price)].copy()

if hide_minutes_risk and "risk_label" in filtered.columns:
    filtered = filtered[filtered["risk_label"] != "Minutes risk"]

if fixture_mode == "Double GW" and "fixture_count" in filtered.columns:
    filtered = filtered[filtered["fixture_count"] >= 2]
elif fixture_mode == "Single GW" and "fixture_count" in filtered.columns:
    filtered = filtered[filtered["fixture_count"] < 2]

filtered = filtered.sort_values("predicted_points", ascending=False).head(top_n)

st.subheader(f"Top {len(filtered)} Players · GW{int(df['gameweek'].max())}")

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
    "Position",
    "Price (£m)",
    "Predicted Pts",
    *[col.replace("_", " ").title() for col in display_cols[5:]],
]
display["Price (£m)"] = display["Price (£m)"].map("{:.1f}".format)
display["Predicted Pts"] = display["Predicted Pts"].map("{:.2f}".format)
if "Value Score" in display.columns:
    display["Value Score"] = display["Value Score"].map("{:.2f}".format)
display = display.reset_index(drop=True)
display.index += 1

st.dataframe(display, use_container_width=True)

st.markdown("---")
st.subheader("Predicted Points Distribution")

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
    height=max(400, len(filtered) * 25),
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3"),
)

st.plotly_chart(fig, use_container_width=True)
