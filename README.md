# ⚽ FPL Intelligence Platform

A live Fantasy Premier League analytics and prediction platform powered by machine learning, deployed with an automated data pipeline.

🔗 **[Live Demo](https://fpl-intelligence-platform.streamlit.app)**

---

## Overview

The FPL Intelligence Platform helps Fantasy Premier League managers make better decisions using data. It pulls live data from the FPL API every gameweek, trains an XGBoost model to predict player points, and serves the results through an interactive Streamlit dashboard — all automatically, with no manual intervention required.

---

## Features

| Feature | Description |
|---|---|
| 🎯 GW Picks | Top predicted players ranked by expected points, filterable by position and budget |
| 🔥 Form Heatmap | Visual breakdown of player form and ICT index trends |
| 📅 Fixture Difficulty | Colour-coded FDR heatmap with opponent names for the next 6 gameweeks |
| 🧠 Squad Optimizer | Builds the highest-scoring 15-player squad within budget using linear programming |
| 👑 Captain Picks | Captain recommendations ranked by predicted points with doubled score shown |
| 👤 My Team | Enter your FPL Team ID to get personalised transfer recommendations |
| 💰 Price Changes | Identifies players at risk of price rise or fall based on transfer activity |

---

## Architecture

```
FPL API ──────────────────────────────────────────────────┐
                                                          ▼
                                               etl/fetch_data.py
                                                          │
                                                          ▼
                                             etl/process_data.py
                                           (feature engineering)
                                                          │
                                                          ▼
                                              etl/update_store.py
                                                          │
                                                          ▼
                                                    Supabase DB
                                                    (PostgreSQL)
                                                    /         \
                                         model/train.py    app/app.py
                                         model/predict.py  (Streamlit)
                                                    \         /
                                                   Supabase DB
```

**Automated via GitHub Actions** — runs every Friday at midnight UTC after the gameweek deadline closes.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Source | [FPL API](https://fantasy.premierleague.com/api/) |
| ETL Pipeline | Python, Pandas, Requests |
| Storage | Supabase (PostgreSQL) |
| Machine Learning | XGBoost, Scikit-learn |
| Optimisation | PuLP (Linear Programming) |
| Scheduling | GitHub Actions (cron) |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |

---

## Model

- **Algorithm**: XGBoost Regressor
- **Target**: `points_per_game` (proxy for expected gameweek points)
- **Features**: Form, ICT index, influence, creativity, threat, minutes, goals, assists, clean sheets, price, transfer activity, position, team
- **Filtering**: Players with fewer than 45 × current gameweek minutes are excluded to remove rotation risks and cup-only players
- **Evaluation**: MAE ~1.8 points, R² ~0.4 (realistic for FPL given inherent randomness)

---

## Project Structure

```
fpl-intelligence-platform/
├── .github/
│   └── workflows/
│       └── etl_pipeline.yml    # GitHub Actions scheduler
├── app/
│   ├── app.py                  # Home page
│   ├── utils.py                # Shared utilities and Supabase connection
│   └── pages/
│       ├── 1_GW_Picks.py
│       ├── 2_Form_Heatmap.py
│       ├── 3_Fixture_Difficulty.py
│       ├── 4_Squad_Optimizer.py
│       ├── 5_Price_Changes.py
│       ├── 6_My_Team.py
│       └── 7_Captain_Pick.py
├── etl/
│   ├── fetch_data.py           # Pulls data from FPL API
│   ├── process_data.py         # Cleans and engineers features
│   └── update_store.py         # Pushes to Supabase
├── model/
│   ├── train.py                # Trains XGBoost model
│   └── predict.py              # Generates predictions and pushes to Supabase
├── .streamlit/
│   └── config.toml             # Streamlit theme config
├── .python-version             # Pins Python 3.11 for Streamlit Cloud
└── requirements.txt
```

---

## How It Works

1. **Every Friday at midnight UTC**, GitHub Actions spins up a Ubuntu VM
2. It runs `fetch_data.py` — pulls bootstrap, fixture, and live GW data from the FPL API
3. Then `process_data.py` — cleans data and engineers features, saves as Parquet
4. Then `update_store.py` — pushes processed player stats to Supabase
5. Then `train.py` — trains a fresh XGBoost model on current season data
6. Then `predict.py` — generates predicted points for all eligible players, pushes to Supabase
7. The Streamlit dashboard reads from Supabase and serves live results to users

---

## Local Setup

```bash
# Clone the repo
git clone https://github.com/Bachu33/fpl-intelligence-platform.git
cd fpl-intelligence-platform

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Fill in your SUPABASE_URL and SUPABASE_KEY in .env

# Run the ETL pipeline
python etl/fetch_data.py
python etl/process_data.py
python etl/update_store.py
python model/train.py
python model/predict.py

# Launch the dashboard
cd app
streamlit run app.py
```

---

## Environment Variables

Create a `.env` file in the project root:

```
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

---

## Author

**Hussein Abdikadir**
BSc Data Science & AI — KCA University, Nairobi
[LinkedIn](https://www.linkedin.com/in/abdikadirhusseinbachu))

---

## License

MIT
