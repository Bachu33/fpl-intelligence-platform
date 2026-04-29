import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_predictions, POSITION_COLORS, POSITION_ORDER

st.set_page_config(page_title="GW Picks", page_icon="🎯", layout="wide")

from utils import apply_custom_css
apply_custom_css()

st.title("🎯 GW Picks")
st.markdown("Top predicted players for this gameweek. Filter by position and budget to find the best options.")
st.markdown("---")

df = load_predictions()

if df.empty:
    st.warning("No predictions available yet. The pipeline may not have run this gameweek.")
    st.stop()

st.sidebar.header("Filters")

positions = st.sidebar.multiselect(
    "Position",
    options=POSITION_ORDER,
    default=POSITION_ORDER
)

max_price = st.sidebar.slider(
    "Max Price (£m)",
    min_value=4.0,
    max_value=15.0,
    value=15.0,
    step=0.5
)

top_n = st.sidebar.slider(
    "Show Top N Players",
    min_value=5,
    max_value=50,
    value=20,
    step=5
)

filtered = df[
    (df["position"].isin(positions)) &
    (df["price"] <= max_price)
].sort_values("predicted_points", ascending=False).head(top_n)

st.subheader(f"Top {top_n} Players — GW{df['gameweek'].max()}")

col1, col2 = st.columns(2)

for idx, pos in enumerate(POSITION_ORDER):
    pos_df = filtered[filtered["position"] == pos]
    if pos_df.empty:
        continue
    
    display_cols = ["player_name", "team", "price", "predicted_points"]
    pos_display = pos_df[display_cols].copy()
    pos_display.columns = ["Player", "Team", "Price (£m)", "Predicted Pts"]
    pos_display["Price (£m)"] = pos_display["Price (£m)"].map("{:.1f}".format)
    pos_display["Predicted Pts"] = pos_display["Predicted Pts"].map("{:.2f}".format)
    pos_display = pos_display.reset_index(drop=True)
    pos_display.index += 1
    
    with (col1 if idx % 2 == 0 else col2):
        st.markdown(f"#### {pos}")
        st.dataframe(pos_display, use_container_width=True, hide_index=False)

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
    title=f"Top {top_n} Players by Predicted Points"
)

fig.update_layout(
    height=max(400, top_n * 25),
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3")
)


st.plotly_chart(fig, use_container_width=True)