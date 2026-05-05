import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    POSITION_ORDER,
    apply_custom_css,
    load_player_stats,
    load_predictions,
    render_player_card,
    using_local_data,
)

st.set_page_config(
    page_title="FPL Intelligence Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

st.title("⚽ FPL Intelligence Platform")
st.caption("Powered by XGBoost · Updated every gameweek · Data from FPL API")
st.markdown("---")

if using_local_data():
    st.info("Running from local generated data. Supabase is not configured or the package is not installed.")

df = load_predictions()
stats = load_player_stats()

if df.empty:
    st.warning("Pipeline hasn't run yet. No data available.")
    st.stop()

current_gw = int(df["gameweek"].max())
last_updated = df["updated_at"].max() if "updated_at" in df.columns else "Local"

if not stats.empty and "selected_by_percent" in stats.columns:
    df = df.merge(
        stats[["player_id", "selected_by_percent", "form", "minutes"]],
        how="left",
        on="player_id",
        suffixes=("", "_stats"),
    )

if "selected_by_percent" not in df.columns:
    df["selected_by_percent"] = 0
df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0)
df["value_score"] = pd.to_numeric(
    df.get("value_score", df["predicted_points"] / df["price"].replace(0, 1)),
    errors="coerce",
).fillna(0)

captain = df.sort_values("predicted_points", ascending=False).iloc[0]
value_pick = df[df["price"] <= 6.5].sort_values("value_score", ascending=False).head(1)
differential = df[df["selected_by_percent"] <= 10].sort_values("predicted_points", ascending=False).head(1)
dgw_count = int((df["fixture_count"] >= 2).sum()) if "fixture_count" in df.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Gameweek", f"GW {current_gw}")
col2.metric("Players Tracked", len(df))
col3.metric("Double-GW Players", dgw_count)
col4.metric("Last Updated", str(last_updated)[:10])

st.markdown("---")
st.subheader("Gameweek Focus")

focus_cols = st.columns(3)
with focus_cols[0]:
    st.markdown(f"""
    <div class="insight-strip">
        <div class="insight-label">Captain</div>
        <div class="insight-value">{captain['player_name']} · {captain['predicted_points']:.2f} pts</div>
    </div>
    """, unsafe_allow_html=True)

with focus_cols[1]:
    if not value_pick.empty:
        pick = value_pick.iloc[0]
        st.markdown(f"""
        <div class="insight-strip">
            <div class="insight-label">Best Value</div>
            <div class="insight-value">{pick['player_name']} · {pick['value_score']:.2f} pts/£m</div>
        </div>
        """, unsafe_allow_html=True)

with focus_cols[2]:
    if not differential.empty:
        pick = differential.iloc[0]
        st.markdown(f"""
        <div class="insight-strip">
            <div class="insight-label">Differential</div>
            <div class="insight-value">{pick['player_name']} · {pick['selected_by_percent']:.1f}% owned</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Top Pick Per Position")

cols = st.columns(4)
for i, pos in enumerate(POSITION_ORDER):
    pos_df = df[df["position"] == pos].sort_values("predicted_points", ascending=False)
    if pos_df.empty:
        continue
    with cols[i]:
        render_player_card(pos_df.iloc[0], rank=pos)

st.markdown("---")
st.subheader("Overall Top Picks")

top5 = df.sort_values("predicted_points", ascending=False).head(5)
display_cols = ["player_name", "team", "position", "price", "predicted_points"]
for optional in ["fixture_label", "opponent", "fdr", "risk_label", "selected_by_percent"]:
    if optional in top5.columns:
        display_cols.append(optional)

top5_display = top5[display_cols].copy()
top5_display.columns = [
    "Player",
    "Team",
    "Position",
    "Price (£m)",
    "Predicted Pts",
    *[col.replace("_", " ").title() for col in display_cols[5:]],
]
top5_display["Price (£m)"] = top5_display["Price (£m)"].map("{:.1f}".format)
top5_display["Predicted Pts"] = top5_display["Predicted Pts"].map("{:.2f}".format)
if "Selected By Percent" in top5_display.columns:
    top5_display["Selected By Percent"] = top5_display["Selected By Percent"].map("{:.1f}%".format)
top5_display = top5_display.reset_index(drop=True)
top5_display.index += 1

st.dataframe(top5_display, use_container_width=True)

st.markdown("---")
st.subheader("Explore")

c1, c2, c3, c4, c5 = st.columns(5)
c1.page_link("pages/1_GW_Picks.py", label="🎯 GW Picks", use_container_width=True)
c2.page_link("pages/2_Form_Heatmap.py", label="🔥 Form Heatmap", use_container_width=True)
c3.page_link("pages/3_Fixture_Difficulty.py", label="📅 Fixtures", use_container_width=True)
c4.page_link("pages/4_Squad_Optimizer.py", label="🧠 Optimizer", use_container_width=True)
c5.page_link("pages/5_Price_Changes.py", label="💰 Price Changes", use_container_width=True)
