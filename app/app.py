import streamlit as st

st.set_page_config(
    page_title="FPL Intelligence Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚽ FPL Intelligence Platform")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("### 🎯 GW Picks\nTop predicted players for this gameweek filtered by position and budget.")

with col2:
    st.info("### 🔥 Form Heatmap\nVisualise which players are in form and which have gone cold.")

with col3:
    st.info("### 📅 Fixture Difficulty\nSee which teams have the easiest runs coming up.")

col4, col5 = st.columns(2)

with col4:
    st.info("### 🧠 Squad Optimizer\nBuild the highest-scoring squad within your budget using linear programming.")

with col5:
    st.info("### 💰 Price Changes\nSpot players about to rise or fall in price before it happens.")

st.markdown("---")
st.caption("Data updates automatically every gameweek via GitHub Actions. Predictions powered by XGBoost.")