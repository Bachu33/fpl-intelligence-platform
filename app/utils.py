import streamlit as st
from supabase import create_client
import pandas as pd
import os
from dotenv import load_dotenv

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
        st.error("Supabase credentials not found.")
        st.stop()
    
    return create_client(url, key)

@st.cache_data(ttl=3600)
def load_predictions():
    supabase = get_supabase_client()
    response = supabase.table("predictions").select("*").execute()
    df = pd.DataFrame(response.data)
    if df.empty:
        return df
    df["predicted_points"] = pd.to_numeric(df["predicted_points"], errors="coerce")
    df["now_cost"] = pd.to_numeric(df["now_cost"], errors="coerce")
    df["price"] = df["now_cost"] / 10
    return df

@st.cache_data(ttl=3600)
def load_player_stats():
    supabase = get_supabase_client()
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
