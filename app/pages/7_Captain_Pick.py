import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_predictions, apply_custom_css, POSITION_COLORS, POSITION_ORDER

st.set_page_config(page_title="Captain Pick", page_icon="👑", layout="wide")
apply_custom_css()

st.title("👑 Captain Recommendations")
st.markdown("Captain the player with the highest predicted points. Your captain scores double — pick right.")
st.markdown("---")

df = load_predictions()

if df.empty:
    st.warning("No predictions available yet.")
    st.stop()

st.sidebar.header("Filters")
max_price = st.sidebar.slider("Max Price (£m)", 4.0, 15.0, 15.0, 0.5)
positions = st.sidebar.multiselect("Position", POSITION_ORDER, default=["MID", "FWD"])

filtered = df[
    (df["position"].isin(positions)) &
    (df["price"] <= max_price)
].sort_values("predicted_points", ascending=False).head(10)

st.subheader("Top 10 Captain Candidates")

for i, (_, row) in enumerate(filtered.iterrows()):
    rank = i + 1
    doubled = round(row["predicted_points"] * 2, 2)
    
    if rank == 1:
        label = "👑 Captain Pick"
        border_color = "#FFD700"
    elif rank == 2:
        label = "🔰 Vice Captain"
        border_color = "#C0C0C0"
    else:
        label = f"#{rank}"
        border_color = "#30363d"
    
    st.markdown(f"""
    <div style="
        background-color: var(--secondary-background-color);
        border: 1px solid {border_color};
        border-left: 5px solid {POSITION_COLORS.get(row['position'], '#00cc6a')};
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div>
            <span style="color:{border_color};font-weight:700;margin-right:12px">{label}</span>
            <span style="font-weight:700;font-size:1.05rem">{row['player_name']}</span>
            <span style="color:#8b949e;margin-left:8px">{row['team']} · {row['position']} · £{row['price']:.1f}m</span>
        </div>
        <div style="text-align:right">
            <div style="color:#00cc6a;font-weight:700;font-size:1.1rem">{row['predicted_points']:.2f} pts</div>
            <div style="color:#FFD700;font-size:0.85rem">2x → {doubled} pts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Predicted Points Comparison")

fig = px.bar(
    filtered.sort_values("predicted_points", ascending=True),
    x="predicted_points",
    y="player_name",
    color="position",
    color_discrete_map=POSITION_COLORS,
    orientation="h",
    labels={"predicted_points": "Predicted Points", "player_name": "Player"},
    hover_data=["team", "price"]
)

fig.update_layout(
    height=450,
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3")
)

st.plotly_chart(fig, use_container_width=True)

st.caption("Predictions are based on the XGBoost model trained on current season data. Always consider upcoming fixtures before captaining.")