# FPL Intelligence Platform

A live Fantasy Premier League prediction and analytics platform.

## Live Demo
[Coming soon]

## What It Does
- Predicts expected points for all FPL players each gameweek using XGBoost
- Recommends optimal squads using linear programming (PuLP)
- Tracks player form, fixture difficulty, and price change risk
- Updates automatically every gameweek via a scheduled ETL pipeline

## Stack
- **Data**: FPL API, football-data.co.uk
- **Processing**: Python, Pandas
- **Storage**: Supabase (PostgreSQL)
- **ML**: XGBoost, Scikit-learn
- **Optimization**: PuLP
- **Scheduling**: GitHub Actions
- **Dashboard**: Streamlit

## Architecture
FPL API → ETL Pipeline → Supabase → Streamlit Dashboard

## Setup
(Instructions coming soon)