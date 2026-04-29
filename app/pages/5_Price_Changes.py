import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_player_stats, POSITION_ORDER

st.set_page_config(page_title="Price Changes", page_icon="💰", layout="wide")

from utils import apply_custom_css
apply_custom_css()

st.title("💰 Price Change Tracker")
st.markdown("Players with high transfer activity are likely to change price soon. Buy before they rise, sell before they fall.")
st.markdown("---")

df = load_player_stats()

if df.empty:
    st.warning("No player data available yet.")
    st.stop()

df = df[df["minutes"] > 0].copy()

df["net_transfers"] = df["transfers_in"] - df["transfers_out"]
df["transfer_pressure"] = df["net_transfers"] / df["selected_by_percent"].replace(0, 1)

rising = df.sort_values("net_transfers", ascending=False).head(20)
falling = df.sort_values("net_transfers", ascending=True).head(20)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 Price Rise Candidates")
    st.caption("High net transfers in — likely to rise")
    
    fig_rise = px.bar(
        rising.sort_values("net_transfers", ascending=True),
        x="net_transfers",
        y="player_name",
        color="position",
        orientation="h",
        labels={"net_transfers": "Net Transfers In", "player_name": "Player"},
        hover_data=["team", "price", "form"]
    )
    fig_rise.update_layout(
    height=500,
    showlegend=False,
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3")
    )
    st.plotly_chart(fig_rise, use_container_width=True)

with col2:
    st.subheader("🔴 Price Fall Candidates")
    st.caption("High net transfers out — likely to fall")
    
    falling_plot = falling.copy()
    falling_plot["net_transfers_abs"] = falling_plot["net_transfers"].abs()
    
    fig_fall = px.bar(
        falling_plot.sort_values("net_transfers_abs", ascending=True),
        x="net_transfers_abs",
        y="player_name",
        color="position",
        orientation="h",
        labels={"net_transfers_abs": "Net Transfers Out", "player_name": "Player"},
        hover_data=["team", "price", "form"]
    )
    fig_fall.update_layout(
    height=500,
    showlegend=False,
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3")
    )
    st.plotly_chart(fig_fall, use_container_width=True)

st.markdown("---")
st.subheader("Transfer Activity vs Form")

fig3 = px.scatter(
    df,
    x="net_transfers",
    y="form",
    color="position",
    size="price",
    hover_name="player_name",
    hover_data=["team", "price"],
    labels={"net_transfers": "Net Transfers", "form": "Form Score"},
    title="Transfer Activity vs Form (bubble size = price)"
)
fig3.update_layout(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3")
)
st.plotly_chart(fig3, use_container_width=True)