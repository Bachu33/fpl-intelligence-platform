import streamlit as st
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, value, PULP_CBC_CMD
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_predictions, POSITION_COLORS

st.set_page_config(page_title="Squad Optimizer", page_icon="🧠", layout="wide")

from utils import apply_custom_css
apply_custom_css()

st.title("🧠 Squad Optimizer")
st.markdown("Builds the highest predicted-points starting XI within your budget using linear programming.")
st.markdown("---")

df = load_predictions()

if df.empty:
    st.warning("No predictions available yet.")
    st.stop()

st.sidebar.header("Constraints")

budget = st.sidebar.slider("Budget (£m)", 75.0, 100.0, 83.0, 0.5)

formation_options = ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"]
formation = st.sidebar.selectbox("Formation", formation_options, index=3)

max_per_team = st.sidebar.slider("Max Players Per Team", 1, 3, 3)

if st.button("🔍 Optimise Squad", type="primary"):
    
    def_count, mid_count, fwd_count = [int(x) for x in formation.split("-")]
    gkp_count = 1
    total_starters = 1 + def_count + mid_count + fwd_count  # always 11
    
    players = df.copy()
    players = players.dropna(subset=["predicted_points", "price"])
    players = players[players["price"] > 0]
    players = players.reset_index(drop=True)
    
    prob = LpProblem("FPL_Squad_Optimizer", LpMaximize)
    
    x = [LpVariable(f"player_{i}", cat=LpBinary) for i in range(len(players))]
    
    prob += lpSum(players.loc[i, "predicted_points"] * x[i] for i in range(len(players)))
    
    prob += lpSum(players.loc[i, "price"] * x[i] for i in range(len(players))) <= budget
    
    prob += lpSum(x[i] for i in range(len(players)) if players.loc[i, "position"] == "GKP") == gkp_count + 1
    prob += lpSum(x[i] for i in range(len(players)) if players.loc[i, "position"] == "DEF") == def_count + 1
    prob += lpSum(x[i] for i in range(len(players)) if players.loc[i, "position"] == "MID") == mid_count + 1
    prob += lpSum(x[i] for i in range(len(players)) if players.loc[i, "position"] == "FWD") == fwd_count + 1
    
    teams = players["team"].unique()
    for team in teams:
        team_indices = players[players["team"] == team].index.tolist()
        prob += lpSum(x[i] for i in team_indices) <= max_per_team
    
    prob.solve(PULP_CBC_CMD(msg=0))
    
    selected = players[[value(x[i]) == 1 for i in range(len(players))]].copy()

    if selected.empty:
        st.error("No valid squad found. Try increasing your budget or relaxing constraints.")
    else:
        starters_list = []
        bench_list = []
        
        pos_starter_counts = {"GKP": gkp_count, "DEF": def_count, "MID": mid_count, "FWD": fwd_count}
        
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_players = selected[selected["position"] == pos].sort_values("predicted_points", ascending=False)
            n_starters = pos_starter_counts[pos]
            starters_list.append(pos_players.head(n_starters))
            bench_list.append(pos_players.tail(1))
        
        starters = pd.concat(starters_list)
        bench = pd.concat(bench_list)
        
        total_cost = selected["price"].sum()
        total_predicted = starters["predicted_points"].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Squad Cost", f"£{total_cost:.1f}m")
        col2.metric("Remaining Budget", f"£{budget - total_cost:.1f}m")
        col3.metric("Starting XI Predicted Points", f"{total_predicted:.1f}")
        
        st.markdown("---")
        st.subheader("Starting XI")
    
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        pos_players = starters[starters["position"] == pos]
        if pos_players.empty:
            continue
        st.markdown(f"#### {pos}")
        display = pos_players[["player_name", "team", "price", "predicted_points"]].copy()
        display.columns = ["Player", "Team", "Price (£m)", "Predicted Pts"]
        display["Price (£m)"] = display["Price (£m)"].map("{:.1f}".format)
        display["Predicted Pts"] = display["Predicted Pts"].map("{:.2f}".format)
        display = display.reset_index(drop=True)
        display.index += 1
        st.dataframe(display, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🪑 Bench")
    
    bench_display = bench[["player_name", "team", "position", "price", "predicted_points"]].copy()
    bench_display.columns = ["Player", "Team", "Position", "Price (£m)", "Predicted Pts"]
    bench_display["Price (£m)"] = bench_display["Price (£m)"].map("{:.1f}".format)
    bench_display["Predicted Pts"] = bench_display["Predicted Pts"].map("{:.2f}".format)
    bench_display = bench_display.reset_index(drop=True)
    bench_display.index += 1
    st.dataframe(bench_display, use_container_width=True)