import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv

PREDICTIONS_LOCAL_PATH = "data/processed/latest_predictions.csv"
PLAYER_STATS_LOCAL_PATH = "data/processed/players.parquet"
PREDICTION_FEATURES_LOCAL_PATH = "data/processed/prediction_features.parquet"
BOOTSTRAP_LOCAL_PATH = "data/raw/bootstrap.json"

def using_local_data():
    return get_supabase_client() is None

@st.cache_resource
def get_supabase_client():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        load_dotenv()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        return None
    
    try:
        from supabase import create_client
    except ModuleNotFoundError:
        return None

    return create_client(url, key)

def _load_local_predictions():
    if not os.path.exists(PREDICTIONS_LOCAL_PATH):
        return pd.DataFrame()

    df = pd.read_csv(PREDICTIONS_LOCAL_PATH)
    if df.empty:
        return df

    df["predicted_points"] = pd.to_numeric(df["predicted_points"], errors="coerce")
    df["now_cost"] = pd.to_numeric(df["now_cost"], errors="coerce")
    df["price"] = df["now_cost"] / 10
    return df

def _load_local_player_stats():
    if not os.path.exists(PLAYER_STATS_LOCAL_PATH):
        return pd.DataFrame()

    df = pd.read_parquet(PLAYER_STATS_LOCAL_PATH)
    if df.empty:
        return df

    for col in ["form", "ict_index", "influence", "creativity",
                "threat", "selected_by_percent", "points_per_game"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["price"] = df["now_cost"] / 10
    return df

def _load_team_names():
    if not os.path.exists(BOOTSTRAP_LOCAL_PATH):
        return {}

    with open(BOOTSTRAP_LOCAL_PATH, "r") as f:
        bootstrap = json.load(f)

    return {team["id"]: team["short_name"] for team in bootstrap.get("teams", [])}

def _numeric_series(df, col, default=0):
    if col in df.columns:
        values = df[col]
    else:
        values = pd.Series(default, index=df.index)
    return pd.to_numeric(values, errors="coerce").fillna(default)

def _attach_prediction_context(df):
    if df.empty or not os.path.exists(PREDICTION_FEATURES_LOCAL_PATH):
        return df

    context = pd.read_parquet(PREDICTION_FEATURES_LOCAL_PATH)
    if context.empty:
        return df

    cols = [
        "player_id", "fixture_count", "fdr", "was_home", "opponent_team_id",
        "minutes_lag1", "minutes_roll3", "minutes_roll5", "starts_lag1",
        "starts_roll3", "total_points_roll3", "ict_index_roll3",
        "expected_goal_involvements_roll3",
    ]
    cols = [col for col in cols if col in context.columns]
    context = context[cols].copy()

    df = df.merge(context, how="left", on="player_id")
    team_names = _load_team_names()

    df["fixture_count"] = _numeric_series(df, "fixture_count", 1)
    df["fdr"] = _numeric_series(df, "fdr", 3)
    df["minutes_roll3"] = _numeric_series(df, "minutes_roll3", 0)
    df["starts_roll3"] = _numeric_series(df, "starts_roll3", 0)
    df["value_score"] = df["predicted_points"] / df["price"].where(df["price"] > 0)
    df["fixture_label"] = df["fixture_count"].apply(lambda count: "DGW" if count >= 2 else "SGW")
    df["opponent"] = df.get("opponent_team_id", pd.Series(dtype="float64")).map(team_names).fillna("TBC")
    df["risk_label"] = df.apply(player_risk_label, axis=1)
    return df

@st.cache_data(ttl=3600)
def load_predictions():
    supabase = get_supabase_client()
    if supabase is None:
        return _load_local_predictions()

    response = supabase.table("predictions").select("*").execute()
    df = pd.DataFrame(response.data)
    if df.empty:
        return df
    df["predicted_points"] = pd.to_numeric(df["predicted_points"], errors="coerce")
    df["now_cost"] = pd.to_numeric(df["now_cost"], errors="coerce")
    df["price"] = df["now_cost"] / 10
    return _attach_prediction_context(df)

@st.cache_data(ttl=3600)
def load_player_stats():
    supabase = get_supabase_client()
    if supabase is None:
        return _load_local_player_stats()

    response = supabase.table("player_gameweek_stats").select("*").execute()
    df = pd.DataFrame(response.data)
    if df.empty:
        return df
    for col in ["form", "ict_index", "influence", "creativity", 
                "threat", "selected_by_percent", "points_per_game"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["price"] = df["now_cost"] / 10
    return df

POSITION_COLORS = {
    "GKP": "#ebff00",
    "DEF": "#00ff87",
    "MID": "#05f0ff",
    "FWD": "#ff4c4c"
}

POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]

def player_risk_label(row):
    minutes_roll3 = float(row.get("minutes_roll3", 0) or 0)
    starts_roll3 = float(row.get("starts_roll3", 0) or 0)

    if minutes_roll3 >= 70 and starts_roll3 >= 0.75:
        return "Nailed"
    if minutes_roll3 >= 45:
        return "Managed"
    return "Minutes risk"

def risk_color(label):
    return {
        "Nailed": "#00cc6a",
        "Managed": "#f7b955",
        "Minutes risk": "#ff5c5c",
    }.get(label, "#8b949e")

def fdr_color(fdr):
    try:
        value = float(fdr)
    except (TypeError, ValueError):
        value = 3

    if value <= 2:
        return "#00cc6a"
    if value <= 3:
        return "#f7b955"
    return "#ff5c5c"

def render_player_card(row, rank=None, captain=False):
    position = row.get("position", "")
    position_color = POSITION_COLORS.get(position, "#00cc6a")
    predicted = float(row.get("predicted_points", 0) or 0)
    price = float(row.get("price", 0) or 0)
    fixture_count = float(row.get("fixture_count", 1) or 1)
    fdr = float(row.get("fdr", 3) or 3)
    risk = row.get("risk_label", "Unknown")
    opponent = row.get("opponent", "TBC")
    rank_label = f"#{rank}" if rank else position
    captain_line = f"<div class='card-subtle'>Captain score: <b>{predicted * 2:.2f}</b></div>" if captain else ""
    dgw_class = "chip chip-hot" if fixture_count >= 2 else "chip"

    st.markdown(f"""
    <div class="player-card" style="border-left-color:{position_color}">
        <div class="card-topline">
            <span class="rank-badge">{rank_label}</span>
            <span class="{dgw_class}">{'DGW' if fixture_count >= 2 else 'SGW'}</span>
            <span class="chip" style="border-color:{fdr_color(fdr)}">FDR {fdr:.1f}</span>
            <span class="chip" style="border-color:{risk_color(risk)}">{risk}</span>
        </div>
        <div class="card-name">{row.get('player_name', 'Unknown')}</div>
        <div class="card-subtle">{row.get('team', 'Unknown')} · {position} · £{price:.1f}m · vs {opponent}</div>
        <div class="card-score">{predicted:.2f} pts</div>
        {captain_line}
    </div>
    """, unsafe_allow_html=True)

def apply_custom_css():
    st.markdown("""
        <style>
        :root {
            --background: #0b1117;
            --foreground: #eef4f8;
            --card: #10151c;
            --card-foreground: #f4f7fb;
            --muted: #161d26;
            --muted-foreground: #9aa6b2;
            --border: rgba(230,237,243,0.12);
            --ring: rgba(0,204,106,0.42);
            --surface-1: var(--card);
            --surface-2: var(--muted);
            --border-soft: var(--border);
            --text-soft: #9aa6b2;
            --accent: #00cc6a;
            --accent-foreground: #ffffff;
            --warn: #f7b955;
            --danger: #ff5c5c;
            --radius: 8px;
        }

        /* Metric boxes */
        [data-testid="stMetric"] {
            background-color: var(--surface-1);
            border: 1px solid var(--border-soft);
            border-radius: 8px;
            padding: 16px;
        }

        [data-testid="stMetricValue"] {
            color: #00cc6a;
            font-size: 1.8rem;
            font-weight: 700;
        }

        /* Dataframe */
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border-soft);
            border-radius: 8px;
        }

        /* Titles */
        h1 {
            color: #00cc6a;
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        h2, h3 {
            font-weight: 600;
        }

        /* Buttons */
        .stButton > button {
            background-color: var(--accent);
            color: var(--accent-foreground);
            font-weight: 700;
            border: 1px solid transparent;
            border-radius: 6px;
            min-height: 36px;
            padding: 0.5rem 1rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.22);
            transition: background-color 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
        }

        .stButton > button:hover {
            background-color: #00aa55;
            color: var(--accent-foreground);
            box-shadow: 0 0 0 3px var(--ring);
        }

        .stButton > button[kind="secondary"] {
            background: var(--surface-2);
            color: var(--foreground);
            border-color: var(--border);
        }

        .ui-card,
        .player-card {
            background: var(--surface-1);
            border: 1px solid var(--border-soft);
            border-radius: var(--radius);
            box-shadow: 0 1px 2px rgba(0,0,0,0.22);
        }

        .player-card {
            border-left: 4px solid var(--accent);
            padding: 14px;
            min-height: 156px;
            margin-bottom: 10px;
            transition: border-color 140ms ease, transform 140ms ease, box-shadow 140ms ease;
        }

        .player-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.22);
        }

        .card-topline {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
            margin-bottom: 10px;
        }

        .ui-badge,
        .rank-badge,
        .chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 22px;
            padding: 0 9px;
            border-radius: 6px;
            border: 1px solid var(--border-soft);
            color: #e6edf3;
            font-size: 0.75rem;
            font-weight: 700;
            line-height: 1;
            transition: background-color 140ms ease, border-color 140ms ease, color 140ms ease;
        }

        .rank-badge {
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
        }

        .chip-hot {
            background: rgba(247,185,85,0.14);
            border-color: var(--warn);
            color: #ffd98a;
        }

        .ui-alert {
            background: rgba(22,29,38,0.84);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--foreground);
            padding: 12px 14px;
            font-size: 0.9rem;
        }

        .card-name {
            color: #f4f7fb;
            font-size: 1.1rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 6px;
        }

        .card-subtle {
            color: var(--text-soft);
            font-size: 0.86rem;
            line-height: 1.35;
        }

        .card-score {
            color: var(--accent);
            font-size: 1.6rem;
            font-weight: 850;
            margin-top: 10px;
        }

        .insight-strip {
            background: var(--surface-1);
            border: 1px solid var(--border-soft);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 96px;
        }

        .insight-label {
            color: var(--text-soft);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .insight-value {
            color: #f4f7fb;
            font-size: 1.1rem;
            font-weight: 800;
        }

        .pitch {
            background:
                linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px),
                linear-gradient(180deg, #146c43 0%, #0f5c39 100%);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 8px;
            padding: 18px 12px;
            margin-top: 10px;
        }

        .pitch-row {
            display: grid;
            gap: 10px;
            justify-content: center;
            margin: 12px 0;
        }

        .pitch-player {
            background: rgba(13,17,23,0.82);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 8px;
            padding: 8px;
            min-width: 116px;
            max-width: 148px;
            text-align: center;
        }

        .pitch-name {
            color: #ffffff;
            font-weight: 800;
            font-size: 0.82rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .pitch-meta {
            color: #d8e2eb;
            font-size: 0.74rem;
            margin-top: 3px;
        }

        /* Layout */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        </style>
    """, unsafe_allow_html=True)
