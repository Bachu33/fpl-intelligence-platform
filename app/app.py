import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))
from utils import load_predictions, load_player_stats, apply_custom_css, POSITION_COLORS

st.set_page_config(
    page_title="FPL Intelligence Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

st.title("⚽ FPL Intelligence Platform")
st.caption("Powered by XGBoost · Updated every gameweek · Data from FPL API")
st.markdown("---")

df = load_predictions()
stats = load_player_stats()

if df.empty:
    st.warning("Pipeline hasn't run yet. No data available.")
    st.stop()

current_gw = df["gameweek"].max()
total_players = len(df)
last_updated = df["updated_at"].max() if "updated_at" in df.columns else "Unknown"

col1, col2, col3 = st.columns(3)
col1.metric("Current Gameweek", f"GW {current_gw}")
col2.metric("Players Tracked", total_players)
col3.metric("Last Updated", str(last_updated)[:10] if last_updated != "Unknown" else "Unknown")

st.markdown("---")

st.subheader("🎯 Top Pick Per Position This GW")

cols = st.columns(4)
positions = ["GKP", "DEF", "MID", "FWD"]

for i, pos in enumerate(positions):
    pos_df = df[df["position"] == pos].sort_values("predicted_points", ascending=False)
    if pos_df.empty:
        continue
    top = pos_df.iloc[0]
    with cols[i]:
        st.markdown(f"""
        <div style="
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.2);
            border-top: 3px solid {POSITION_COLORS[pos]};
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        ">
            <div style="color: {POSITION_COLORS[pos]}; font-weight: 700; font-size: 0.85rem;">{pos}</div>
            <div style="font-weight: 700; font-size: 1.1rem; margin: 8px 0;">{top['player_name']}</div>
            <div style="color: #8b949e; font-size: 0.85rem;">{top['team']}</div>
            <div style="color: #00cc6a; font-weight: 700; font-size: 1.4rem; margin-top: 8px;">{top['predicted_points']:.2f} pts</div>
            <div style="color: #8b949e; font-size: 0.8rem;">£{top['price']:.1f}m</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("🏆 Overall Top 5 Picks")

top5 = df.sort_values("predicted_points", ascending=False).head(5)
top5_display = top5[["player_name", "team", "position", "price", "predicted_points"]].copy()
top5_display.columns = ["Player", "Team", "Position", "Price (£m)", "Predicted Pts"]
top5_display["Price (£m)"] = top5_display["Price (£m)"].map("{:.1f}".format)
top5_display["Predicted Pts"] = top5_display["Predicted Pts"].map("{:.2f}".format)
top5_display = top5_display.reset_index(drop=True)
top5_display.index += 1

st.dataframe(top5_display, use_container_width=True)

st.markdown("---")

st.subheader("🔍 Explore")

c1, c2, c3, c4, c5 = st.columns(5)
c1.page_link("pages/1_GW_Picks.py", label="🎯 GW Picks", use_container_width=True)
c2.page_link("pages/2_Form_Heatmap.py", label="🔥 Form Heatmap", use_container_width=True)
c3.page_link("pages/3_Fixture_Difficulty.py", label="📅 Fixtures", use_container_width=True)
c4.page_link("pages/4_Squad_Optimizer.py", label="🧠 Optimizer", use_container_width=True)
c5.page_link("pages/5_Price_Changes.py", label="💰 Price Changes", use_container_width=True)