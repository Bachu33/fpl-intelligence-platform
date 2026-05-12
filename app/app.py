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
    prediction_column_config,
    render_app_header,
    render_kicker,
    render_player_card,
    render_sidebar_nav,
    using_local_data,
)

st.set_page_config(
    page_title="FPL Intelligence Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar_nav("Dashboard")

if using_local_data():
    st.info("Running from local generated data. Supabase is not configured or the package is not installed.")

df = load_predictions()
stats = load_player_stats()

if df.empty:
    st.warning("Pipeline hasn't run yet. No data available.")
    st.stop()

current_gw = int(df["gameweek"].max())
last_updated = df["updated_at"].max() if "updated_at" in df.columns else "Local"

render_kicker(f"Gameweek {current_gw} · Active")
render_app_header(
    "FPL Intelligence Dashboard",
    "Machine-learning powered predictions, value signals, fixture context, and squad decisions refreshed from the official FPL API.",
    badges=[f"GW {current_gw}", "XGBoost", "Rolling backtested", "FPL API"],
)

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

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Gameweek", f"GW {current_gw}", "Live model board")
c2.metric("Players Tracked", f"{len(df):,}", "Premier League pool")
c3.metric("Double-GW Players", dgw_count, "Fixture count >= 2")
c4.metric("Model Last Trained", str(last_updated)[:10], "XGBoost")

st.markdown("### Gameweek Focus")
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

st.markdown("### Top Predicted By Position")
cols = st.columns(4)
for i, pos in enumerate(POSITION_ORDER):
    pos_df = df[df["position"] == pos].sort_values("predicted_points", ascending=False)
    if pos_df.empty:
        continue
    with cols[i]:
        render_player_card(pos_df.iloc[0], rank=pos)

st.markdown("### Top 5 Overall")
top5 = df.sort_values("predicted_points", ascending=False).head(5)
display_cols = ["player_name", "team", "position", "price", "predicted_points"]
for optional in ["fixture_label", "opponent", "fdr", "risk_label", "selected_by_percent"]:
    if optional in top5.columns:
        display_cols.append(optional)

top5_display = top5[display_cols].copy()
top5_display.columns = [
    "Player",
    "Team",
    "Pos",
    "Price (£m)",
    "Predicted Pts",
    *[col.replace("_", " ").title() for col in display_cols[5:]],
]
top5_display.insert(0, "#", range(1, len(top5_display) + 1))

st.dataframe(
    top5_display,
    use_container_width=True,
    hide_index=True,
    column_config=prediction_column_config(),
)
