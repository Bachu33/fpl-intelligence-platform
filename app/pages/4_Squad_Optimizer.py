import os
import sys

import pandas as pd
import streamlit as st
from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD, lpSum, value

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import POSITION_ORDER, apply_custom_css, load_predictions, render_app_header

st.set_page_config(page_title="Squad Optimizer", page_icon="🧠", layout="wide")
apply_custom_css()

st.title("🧠 Squad Optimizer")
st.markdown("---")

df = load_predictions()

if df.empty:
    st.warning("No predictions available yet.")
    st.stop()

render_app_header(
    "Squad Builder",
    "A pitch-first optimizer for testing formations, budget, team limits, and minutes-risk constraints.",
    badges=["Linear programming", "Pitch view", "Bench"],
)

st.sidebar.header("Constraints")
budget = st.sidebar.slider("Budget (£m)", 75.0, 100.0, 83.0, 0.5)
formation_options = ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"]
formation = st.sidebar.selectbox("Formation", formation_options, index=3)
max_per_team = st.sidebar.slider("Max Players Per Team", 1, 3, 3)
avoid_minutes_risk = st.sidebar.checkbox("Avoid minutes-risk players", value=False)


def render_pitch(starters):
    POSITION_COLORS = {"GKP": "#ebff00", "DEF": "#00ff87", "MID": "#05f0ff", "FWD": "#ff4c4c"}

    rows = []
    for pos in POSITION_ORDER:
        pos_players = starters[starters["position"] == pos].sort_values("predicted_points", ascending=False)
        pos_color = POSITION_COLORS.get(pos, "#00ff87")
        cells = []
        for _, player in pos_players.iterrows():
            label = "DGW" if float(player.get("fixture_count", 1) or 1) >= 2 else "SGW"
            cells.append(
                f'<div style="background:rgba(13,17,23,0.85);border:1px solid rgba(255,255,255,0.15);'
                f'border-top:3px solid {pos_color};border-radius:8px;padding:8px 10px;'
                f'min-width:116px;max-width:148px;text-align:center;">'
                f'<div style="color:#ffffff;font-weight:800;font-size:13px;white-space:nowrap;'
                f'overflow:hidden;text-overflow:ellipsis;">{player["player_name"]}</div>'
                f'<div style="color:#d8e2eb;font-size:11px;margin-top:3px;">{player["team"]} · {player["predicted_points"]:.1f}pts</div>'
                f'<div style="color:#d8e2eb;font-size:11px;margin-top:2px;">{label} · £{player["price"]:.1f}m</div>'
                f'</div>'
            )
        if cells:
            rows.append(
                f'<div style="display:flex;justify-content:center;gap:10px;margin:10px 0;">'
                f'{"".join(cells)}'
                f'</div>'
            )

    pitch_html = (
        f'<div style="background:linear-gradient(180deg,#146c43 0%,#0f5c39 100%);'
        f'border:2px solid rgba(255,255,255,0.2);border-radius:12px;'
        f'padding:20px 16px;margin-top:8px;">'
        f'{"".join(rows)}'
        f'</div>'
    )

    st.markdown(pitch_html, unsafe_allow_html=True)


if st.button("🔍 Optimise Squad", type="primary"):
    def_count, mid_count, fwd_count = [int(x) for x in formation.split("-")]
    gkp_count = 1

    players = df.copy()
    players = players.dropna(subset=["predicted_points", "price"])
    players = players[players["price"] > 0]
    if avoid_minutes_risk and "risk_label" in players.columns:
        players = players[players["risk_label"] != "Minutes risk"]
    players = players.reset_index(drop=True)

    prob = LpProblem("FPL_Squad_Optimizer", LpMaximize)
    x = [LpVariable(f"player_{i}", cat=LpBinary) for i in range(len(players))]

    prob += lpSum(players.loc[i, "predicted_points"] * x[i] for i in range(len(players)))
    prob += lpSum(players.loc[i, "price"] * x[i] for i in range(len(players))) <= budget

    prob += lpSum(x[i] for i in range(len(players)) if players.loc[i, "position"] == "GKP") == gkp_count + 1
    prob += lpSum(x[i] for i in range(len(players)) if players.loc[i, "position"] == "DEF") == def_count + 1
    prob += lpSum(x[i] for i in range(len(players)) if players.loc[i, "position"] == "MID") == mid_count + 1
    prob += lpSum(x[i] for i in range(len(players)) if players.loc[i, "position"] == "FWD") == fwd_count + 1

    for team in players["team"].unique():
        team_indices = players[players["team"] == team].index.tolist()
        prob += lpSum(x[i] for i in team_indices) <= max_per_team

    prob.solve(PULP_CBC_CMD(msg=0))

    selected = players[[value(x[i]) == 1 for i in range(len(players))]].copy()

    if selected.empty:
        st.error("No valid squad found. Try increasing your budget or relaxing constraints.")
        st.stop()

    starters_list = []
    bench_list = []
    starter_counts = {"GKP": gkp_count, "DEF": def_count, "MID": mid_count, "FWD": fwd_count}

    for pos in POSITION_ORDER:
        pos_players = selected[selected["position"] == pos].sort_values("predicted_points", ascending=False)
        starters_list.append(pos_players.head(starter_counts[pos]))
        bench_list.append(pos_players.tail(1))

    starters = pd.concat(starters_list)
    bench = pd.concat(bench_list)

    total_cost = selected["price"].sum()
    total_predicted = starters["predicted_points"].sum()
    minutes_risks = int((selected.get("risk_label", pd.Series(dtype=str)) == "Minutes risk").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Squad Cost", f"£{total_cost:.1f}m")
    col2.metric("Remaining", f"£{budget - total_cost:.1f}m")
    col3.metric("Starting XI", f"{total_predicted:.1f} pts")
    col4.metric("Risk Flags", minutes_risks)

    st.markdown("---")
    st.subheader("Starting XI")
    render_pitch(starters)

    st.markdown("---")
    detail_cols = ["player_name", "team", "position", "price", "predicted_points"]
    for optional in ["fixture_label", "opponent", "fdr", "risk_label"]:
        if optional in selected.columns:
            detail_cols.append(optional)

    starters_display = starters[detail_cols].copy()
    starters_display.columns = [
        "Player", "Team", "Position", "Price (£m)", "Predicted Pts",
        *[col.replace("_", " ").title() for col in detail_cols[5:]],
    ]
    starters_display["Price (£m)"] = starters_display["Price (£m)"].map("{:.1f}".format)
    starters_display["Predicted Pts"] = starters_display["Predicted Pts"].map("{:.2f}".format)
    st.dataframe(starters_display.reset_index(drop=True), use_container_width=True)

    st.subheader("Bench")
    bench_display = bench[detail_cols].copy()
    bench_display.columns = starters_display.columns
    bench_display["Price (£m)"] = bench_display["Price (£m)"].map("{:.1f}".format)
    bench_display["Predicted Pts"] = bench_display["Predicted Pts"].map("{:.2f}".format)
    st.dataframe(bench_display.reset_index(drop=True), use_container_width=True)
