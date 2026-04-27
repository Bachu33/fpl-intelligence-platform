import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import pulp
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from difflib import get_close_matches
from io import StringIO

st.set_page_config(page_title="FPL xP Tool", layout="wide", page_icon="⚽")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background-color: #0a0a0a; color: #f0f0f0; }
  section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #222; }
  .stTabs [data-baseweb="tab-list"] { background-color: #111; border-bottom: 1px solid #222; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background-color: #1a1a1a; color: #888; border-radius: 4px 4px 0 0; padding: 8px 16px; font-size: 13px; font-weight: 600; }
  .stTabs [aria-selected="true"] { background-color: #222; color: #fff; border-bottom: 2px solid #fff; }
  .stButton > button { background-color: #fff; color: #000; border: none; font-weight: 700; border-radius: 4px; transition: all 0.2s; }
  .stButton > button:hover { background-color: #ddd; }
  .stDataFrame { background-color: #111; border: 1px solid #222; border-radius: 6px; }
  div[data-testid="metric-container"] { background-color: #111; border: 1px solid #222; border-radius: 6px; padding: 12px 16px; }
  div[data-testid="metric-container"] label { color: #888; font-size: 12px; }
  div[data-testid="metric-container"] div { color: #fff; font-weight: 700; }
  .stExpander { background-color: #111; border: 1px solid #222; border-radius: 6px; }
  .stSelectbox > div, .stTextInput > div > div { background-color: #1a1a1a !important; border: 1px solid #333 !important; color: #fff !important; }
  h1,h2,h3 { color: #fff; font-weight: 700; }
  h4 { color: #aaa; font-weight: 600; }
  .good  { color: #00cc66; font-weight: 700; }
  .bad   { color: #ff4444; font-weight: 700; }
  .warn  { color: #f0a500; font-weight: 700; }
  .muted { color: #666; font-size: 12px; }
  div[data-testid="stProgress"] > div > div { background-color: #fff; }
  hr { border-color: #222; }
</style>
""", unsafe_allow_html=True)

HEADERS = {'User-Agent': 'Mozilla/5.0'}
API     = "https://fantasy.premierleague.com/api"
POS_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

ALL_CHIPS = {
    'wildcard': 'Wildcard',
    'bboost':   'Bench Boost',
    'freehit':  'Free Hit',
    '3xc':      'Triple Captain',
}

# Full feature set — original proven base + home/away splits + minutes trend
# + direct goal/assist rolling + season-aware opponent GC
FEATURE_COLS = [
    'pts_avg_3',    'pts_avg_5',    'pts_avg_10',
    'xg_avg_3',     'xg_avg_5',     'xg_avg_10',
    'xa_avg_3',     'xa_avg_5',     'xa_avg_10',
    'xgi_avg_3',    'xgi_avg_5',
    'mins_avg_3',   'mins_avg_5',
    'bonus_avg_3',  'bonus_avg_5',
    'goals_avg_3',  'goals_avg_5',
    'assists_avg_3',
    'pts_home_avg5', 'pts_away_avg5',
    'minutes_ratio', 'mins_trend',
    'is_home',
    'opp_attack_norm', 'opp_defence_norm',
    'opp_gc_rolling5',
    'price_norm', 'cs_rolling5',
]

DIFF_MULT           = {1: 1.4, 2: 1.2, 3: 1.0, 4: 0.5, 5: 0.2}
BASELINE_GOAL_PROB  = {'FWD': 0.15, 'MID': 0.08, 'DEF': 0.03, 'GKP': 0.0}

# Seasons to pull from vaastav — weighted so recent matters more
HISTORICAL_SEASONS = {
    '2021-22': 0.5,
    '2022-23': 0.7,
    '2023-24': 1.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# FETCHING — FPL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bootstrap():
    data = requests.get(f"{API}/bootstrap-static/", headers=HEADERS).json()
    return pd.DataFrame(data['elements']), pd.DataFrame(data['teams'])

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fixtures():
    return pd.DataFrame(requests.get(f"{API}/fixtures/", headers=HEADERS).json())

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_histories(eligible_ids):
    all_history = []
    bar = st.progress(0, text="Fetching player histories...")
    for i, pid in enumerate(eligible_ids):
        r = requests.get(f"{API}/element-summary/{pid}/", headers=HEADERS)
        if r.status_code == 200:
            hist = r.json().get('history', [])
            if hist:
                df = pd.DataFrame(hist)
                df['player_id'] = pid
                all_history.append(df)
        if i % 50 == 0:
            bar.progress(i / len(eligible_ids),
                         text=f"Fetching histories... {i}/{len(eligible_ids)}")
            time.sleep(0.2)
    bar.empty()
    return pd.concat(all_history, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# FETCHING — Historical data from vaastav's GitHub repo
# 3 past seasons used for training only — predictions still use current season
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_historical_data():
    base = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    all_dfs = []
    for season, weight in HISTORICAL_SEASONS.items():
        url = f"{base}/{season}/gws/merged_gw.csv"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            df = pd.read_csv(StringIO(r.text))

            # Normalise column names
            df.columns = [c.strip().lower() for c in df.columns]
            if 'gw' in df.columns and 'round' not in df.columns:
                df = df.rename(columns={'gw': 'round'})

            # String player_id — name is enough for groupby rolling
            df['player_id'] = df['name'].astype(str) + '_hist'
            df['season_weight'] = weight

            # Ensure numeric
            num_cols = ['total_points', 'minutes', 'goals_scored', 'assists',
                        'clean_sheets', 'bonus', 'expected_goals', 'expected_assists',
                        'expected_goal_involvements', 'was_home', 'value',
                        'team_h_score', 'team_a_score', 'round']
            for col in num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else:
                    df[col] = 0.0

            # Normalise position labels
            if 'position' not in df.columns and 'element_type' in df.columns:
                df['position'] = df['element_type'].map(POS_MAP)

            all_dfs.append(df)
        except Exception:
            continue

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# ODDS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_odds(api_key):
    if not api_key or not api_key.strip():
        return {}, {}
    try:
        events_r = requests.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/events",
            params={'apiKey': api_key}, timeout=10)
        if events_r.status_code != 200:
            return {}, {}
        events = events_r.json()

        scorer_probs, cs_probs = {}, {}
        for event in events[:10]:
            event_id  = event['id']
            home_team = event.get('home_team', '').lower()
            away_team = event.get('away_team', '').lower()

            props_r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/soccer_epl/events/{event_id}/odds",
                params={'apiKey': api_key, 'regions': 'uk,eu',
                        'markets': 'player_goal_scorer_anytime',
                        'oddsFormat': 'decimal'}, timeout=10)
            if props_r.status_code == 200:
                player_odds = {}
                for bk in props_r.json().get('bookmakers', []):
                    for mk in bk.get('markets', []):
                        if mk['key'] == 'player_goal_scorer_anytime':
                            for o in mk.get('outcomes', []):
                                player_odds.setdefault(o['name'].lower().strip(), []).append(float(o['price']))
                for player, odds_list in player_odds.items():
                    scorer_probs[player] = round((1 / np.mean(odds_list)) * 0.94, 4)

            totals_r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/soccer_epl/events/{event_id}/odds",
                params={'apiKey': api_key, 'regions': 'uk,eu',
                        'markets': 'totals', 'oddsFormat': 'decimal'}, timeout=10)
            if totals_r.status_code == 200:
                under_05 = []
                for bk in totals_r.json().get('bookmakers', []):
                    for mk in bk.get('markets', []):
                        if mk['key'] == 'totals':
                            for o in mk.get('outcomes', []):
                                if o['name'] == 'Under' and float(o.get('point', 99)) <= 0.5:
                                    under_05.append(float(o['price']))
                if under_05:
                    both_cs = (1 / np.mean(under_05)) * 0.94
                    cs_probs[home_team] = round(both_cs ** 0.5, 4)
                    cs_probs[away_team] = round(both_cs ** 0.5, 4)

        return scorer_probs, cs_probs
    except Exception:
        return {}, {}

def match_odds_names(fpl_names, odds_names):
    mapping  = {}
    odds_low = [n.lower() for n in odds_names]
    for name in fpl_names:
        matches = get_close_matches(name.lower(), odds_low, n=1, cutoff=0.6)
        if matches:
            mapping[name] = matches[0]
        else:
            for on in odds_low:
                if name.lower() in on.split():
                    mapping[name] = on
                    break
    return mapping

def apply_odds_multiplier(xp, pos, player_name, scorer_probs, cs_probs,
                           team_name, odds_name_map):
    multiplier = 1.0
    if pos in ['FWD', 'MID']:
        baseline = BASELINE_GOAL_PROB.get(pos, 0.1)
        odds_key = odds_name_map.get(player_name)
        if odds_key and odds_key in scorer_probs and baseline > 0:
            multiplier = float(np.clip(scorer_probs[odds_key] / baseline, 0.75, 1.35))
    elif pos in ['DEF', 'GKP']:
        team_key = team_name.lower() if team_name else ''
        if team_key not in cs_probs:
            m = get_close_matches(team_key, list(cs_probs.keys()), n=1, cutoff=0.5)
            team_key = m[0] if m else ''
        if team_key in cs_probs:
            multiplier = float(np.clip(cs_probs[team_key] / 0.28, 0.75, 1.35))
    return round(xp * multiplier, 2), round(multiplier, 3)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES — single function handles both current and historical data
# fpl_teams=None → use neutral opponent values (for historical seasons)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df, fpl_teams=None):
    df = df.copy().sort_values(['player_id', 'round'])

    roll_cols = [
        ('total_points',               'pts'),
        ('expected_goals',             'xg'),
        ('expected_assists',           'xa'),
        ('expected_goal_involvements', 'xgi'),
        ('minutes',                    'mins'),
        ('bonus',                      'bonus'),
        ('goals_scored',               'goals'),
        ('assists',                    'assists'),
    ]
    for window in [3, 5, 10]:
        for col, feat in roll_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[f'{feat}_avg_{window}'] = df.groupby('player_id')[col].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
            else:
                df[f'{feat}_avg_{window}'] = 0.0

    # Home / away split — key signal for players with strong venue bias
    df['_pts_home'] = df['total_points'].where(df['was_home'].astype(bool), np.nan)
    df['_pts_away'] = df['total_points'].where(~df['was_home'].astype(bool), np.nan)
    df['pts_home_avg5'] = df.groupby('player_id')['_pts_home'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['pts_away_avg5'] = df.groupby('player_id')['_pts_away'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    # Fill NaN (player only played home/away recently) with overall avg
    df['pts_home_avg5'] = df['pts_home_avg5'].fillna(df['pts_avg_5'])
    df['pts_away_avg5'] = df['pts_away_avg5'].fillna(df['pts_avg_5'])
    df.drop(columns=['_pts_home', '_pts_away'], inplace=True)

    # Minutes trend — positive = getting more minutes (nailed), negative = rotation risk
    df['minutes_ratio'] = (df['mins_avg_5'] / 90).clip(0, 1)
    df['mins_trend']    = (df['mins_avg_3'] - df['mins_avg_10']).fillna(0)
    df['is_home']       = df['was_home'].astype(int)

    # Opponent strength — full computation if fpl_teams provided
    if fpl_teams is not None:
        ts = fpl_teams.set_index('id')[[
            'strength_attack_home', 'strength_attack_away',
            'strength_defence_home', 'strength_defence_away'
        ]]
        df['opp_attack_strength'] = df.apply(
            lambda r: ts.loc[r['opponent_team'], 'strength_attack_home']
            if not r['was_home'] and r['opponent_team'] in ts.index
            else ts.loc[r['opponent_team'], 'strength_attack_away']
            if r['opponent_team'] in ts.index else np.nan, axis=1)
        df['opp_defence_strength'] = df.apply(
            lambda r: ts.loc[r['opponent_team'], 'strength_defence_home']
            if r['was_home'] and r['opponent_team'] in ts.index
            else ts.loc[r['opponent_team'], 'strength_defence_away']
            if r['opponent_team'] in ts.index else np.nan, axis=1)
        for col, norm in [('opp_attack_strength', 'opp_attack_norm'),
                          ('opp_defence_strength', 'opp_defence_norm')]:
            mn, mx = df[col].min(), df[col].max()
            df[norm] = (df[col] - mn) / (mx - mn + 1e-9)
    else:
        # Historical data — no team strength available, use neutral
        df['opp_attack_norm']  = 0.5
        df['opp_defence_norm'] = 0.5
        df['opp_attack_strength']  = np.nan
        df['opp_defence_strength'] = np.nan

    # Opponent GC rolling — computable from score columns in both current + historical
    score_col = next((c for c in ['team_h_score', 'team_a_score'] if c in df.columns), None)
    if score_col and 'opponent_team' in df.columns:
        team_gc = df.groupby(['opponent_team', 'round'])[score_col].mean().reset_index()
        team_gc.columns = ['_tid', 'round', 'gsa']
        team_gc['opp_gc_rolling5'] = team_gc.groupby('_tid')['gsa'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df = df.merge(team_gc[['_tid', 'round', 'opp_gc_rolling5']],
                      left_on=['opponent_team', 'round'],
                      right_on=['_tid', 'round'], how='left').drop(columns='_tid')
        df['opp_gc_rolling5'] = df['opp_gc_rolling5'].fillna(0.5)
    else:
        df['opp_gc_rolling5'] = 0.5

    df['price_norm']   = ((df['value'] / 10) - 3.5) / (15 - 3.5) if 'value' in df.columns else 0.5
    df['cs_rolling5']  = df.groupby('player_id')['clean_sheets'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ).fillna(0) if 'clean_sheets' in df.columns else 0.0
    df['volatility_5'] = df.groupby('player_id')['total_points'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).std())
    df['goals_avg_5']  = df['goals_avg_5']  if 'goals_avg_5'  in df.columns else 0.0

    df['target'] = df['total_points']
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MODEL — trains on current + historical combined, weighted by season recency
# ─────────────────────────────────────────────────────────────────────────────
def train_position_models(current_features, historical_features=None):
    models, maes = {}, {}

    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        # Current season data
        curr = current_features[current_features['position'] == pos].dropna(
            subset=FEATURE_COLS + ['target']).sort_values('round').copy()
        curr['sample_weight'] = 1.5  # current season weighted highest

        frames = [curr]

        # Historical data for this position
        if historical_features is not None and not historical_features.empty:
            hist_pos = historical_features[
                historical_features['position'] == pos
            ].dropna(subset=FEATURE_COLS + ['target']).copy()
            if not hist_pos.empty:
                hist_pos['sample_weight'] = hist_pos.get('season_weight', 0.7)
                frames.append(hist_pos)

        combined = pd.concat(frames, ignore_index=True)
        if len(combined) < 100:
            continue

        # Time-based split on current season only for evaluation
        split_gw = int(curr['round'].quantile(0.8))
        train    = combined[~((combined['round'] >= split_gw) &
                               (combined['sample_weight'] == 1.5))]
        test     = curr[curr['round'] >= split_gw]

        if len(train) < 50 or len(test) < 10:
            continue

        model = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=42, verbosity=0
        )
        model.fit(
            train[FEATURE_COLS].fillna(0), train['target'],
            sample_weight=train['sample_weight'],
            eval_set=[(test[FEATURE_COLS].fillna(0), test['target'])],
            verbose=False
        )
        preds       = model.predict(test[FEATURE_COLS].fillna(0))
        models[pos] = model
        maes[pos]   = round(mean_absolute_error(test['target'], preds), 3)

    return models, maes

# ─────────────────────────────────────────────────────────────────────────────
# FORM MULTIPLIER
# ─────────────────────────────────────────────────────────────────────────────
def apply_form_multiplier(xp, pts_avg_3, pts_avg_10):
    if pd.isna(pts_avg_3) or pd.isna(pts_avg_10) or pts_avg_10 < 0.5:
        return xp
    return round(xp * float(np.clip(pts_avg_3 / pts_avg_10, 0.6, 1.4)), 2)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_fixture_for_gw(team_id, gw, future):
    home = future[(future['team_h'] == team_id) & (future['event'] == gw)].head(1)
    away = future[(future['team_a'] == team_id) & (future['event'] == gw)].head(1)
    if not home.empty:
        r = home.iloc[0]
        return {'is_home': True,  'opponent': r['team_a'], 'difficulty': int(r['team_h_difficulty'])}
    if not away.empty:
        r = away.iloc[0]
        return {'is_home': False, 'opponent': r['team_h'], 'difficulty': int(r['team_a_difficulty'])}
    return None

def predict_for_gw(player_row, fix, model, fpl_teams, all_att, all_def):
    ts   = fpl_teams.set_index('id')
    feat = player_row[FEATURE_COLS].copy()
    feat['is_home'] = int(fix['is_home'])
    opp_id = fix['opponent']
    if opp_id in ts.index:
        opp     = ts.loc[opp_id]
        raw_att = opp['strength_attack_home']  if not fix['is_home'] else opp['strength_attack_away']
        raw_def = opp['strength_defence_home'] if fix['is_home']     else opp['strength_defence_away']
        feat['opp_attack_norm']  = (raw_att - all_att.min()) / (all_att.max() - all_att.min() + 1e-9)
        feat['opp_defence_norm'] = (raw_def - all_def.min()) / (all_def.max() - all_def.min() + 1e-9)
        # Override home/away pts with correct context
        feat['pts_home_avg5'] = player_row.get('pts_home_avg5', player_row.get('pts_avg_5', 0))
        feat['pts_away_avg5'] = player_row.get('pts_away_avg5', player_row.get('pts_avg_5', 0))
    return max(0, round(float(model.predict(
        pd.DataFrame([feat])[FEATURE_COLS].fillna(0))[0]), 2))

def predict_next_gw(features_df, models, fpl_players, fpl_teams, fixtures,
                    scorer_probs, cs_probs, n_gws=3):
    future    = fixtures[fixtures['finished'] == False].copy()
    next_gw   = int(future['event'].dropna().min())
    ts        = fpl_teams.set_index('id')
    all_att   = features_df['opp_attack_strength'].dropna()
    all_def   = features_df['opp_defence_strength'].dropna()
    latest    = features_df.sort_values('round').groupby('player_id').last().reset_index()
    gws_ahead = sorted(future['event'].dropna().unique())[:n_gws]

    fpl_web_names = fpl_players['web_name'].tolist()
    odds_name_map = match_odds_names(fpl_web_names, list(scorer_probs.keys())) \
                    if scorer_probs else {}
    odds_available = bool(scorer_probs or cs_probs)

    def get_next_5(team_id):
        home = future[future['team_h'] == team_id][['event', 'team_a', 'team_h_difficulty']].copy()
        home.columns = ['event', 'opponent', 'difficulty']; home['is_home'] = True
        away = future[future['team_a'] == team_id][['event', 'team_h', 'team_a_difficulty']].copy()
        away.columns = ['event', 'opponent', 'difficulty']; away['is_home'] = False
        all_f = pd.concat([home, away]).sort_values('event').head(5)
        return ' | '.join(
            f"{'H' if r['is_home'] else 'A'} {ts.loc[r['opponent'],'short_name'] if r['opponent'] in ts.index else '?'}[{int(r['difficulty'])}]"
            for _, r in all_f.iterrows()
        )

    rows = []
    for _, player in latest.iterrows():
        pid, pos = player['player_id'], player['position']
        if pos not in models:
            continue
        fpl_row = fpl_players[fpl_players['id'] == pid]
        if fpl_row.empty:
            continue
        fpl_row   = fpl_row.iloc[0]
        team_id   = int(fpl_row['team'])
        web_name  = fpl_row['web_name']
        team_name = fpl_row.get('team_name', '')

        fix1 = get_fixture_for_gw(team_id, next_gw, future)
        if fix1 is None:
            continue

        # 1. Model prediction
        raw_xp = predict_for_gw(player, fix1, models[pos], fpl_teams, all_att, all_def)

        # 2. Form multiplier
        pts_avg_3  = player.get('pts_avg_3',  np.nan)
        pts_avg_10 = player.get('pts_avg_10', np.nan)
        xp_form    = apply_form_multiplier(raw_xp, pts_avg_3, pts_avg_10)

        # 3. Odds multiplier
        if odds_available:
            xp1, odds_mult = apply_odds_multiplier(
                xp_form, pos, web_name, scorer_probs, cs_probs, team_name, odds_name_map)
        else:
            xp1, odds_mult = xp_form, 1.0

        # Multi-GW
        multi_xps = []
        for gw in gws_ahead:
            fix = get_fixture_for_gw(team_id, gw, future)
            if fix:
                r   = predict_for_gw(player, fix, models[pos], fpl_teams, all_att, all_def)
                frm = apply_form_multiplier(r, pts_avg_3, pts_avg_10)
                if odds_available:
                    fin, _ = apply_odds_multiplier(frm, pos, web_name, scorer_probs,
                                                   cs_probs, team_name, odds_name_map)
                else:
                    fin = frm
                multi_xps.append(fin)
        multi_xp_avg = round(np.mean(multi_xps), 2) if multi_xps else xp1

        vol        = player.get('volatility_5', 0)
        vol        = 0 if pd.isna(vol) else round(float(vol), 2)
        goals_avg5 = player.get('goals_avg_5',  0)
        goals_avg5 = 0 if pd.isna(goals_avg5) else float(goals_avg5)
        mins_trend = player.get('mins_trend',   0)
        mins_trend = 0 if pd.isna(mins_trend) else float(mins_trend)

        opp_id     = fix1['opponent']
        opp_short  = ts.loc[opp_id, 'short_name'] if opp_id in ts.index else '?'
        difficulty = fix1['difficulty']
        diff_mult  = DIFF_MULT.get(difficulty, 1.0)

        is_def_scorer = (pos == 'DEF' and goals_avg5 >= 0.15)
        if pos in ['MID', 'FWD'] or is_def_scorer:
            cap_score = max(0, round((xp1 + 0.3 * vol) * diff_mult, 2))
        else:
            cap_score = 0.0

        # Form tag
        if not pd.isna(pts_avg_3) and not pd.isna(pts_avg_10) and pts_avg_10 >= 0.5:
            ratio    = pts_avg_3 / pts_avg_10
            form_tag = '🔥' if ratio >= 1.15 else '❄️' if ratio <= 0.75 else '➡️'
        else:
            form_tag = '➡️'

        # Odds tag
        odds_tag = '📈' if odds_mult > 1.05 else '📉' if odds_mult < 0.95 else '➖'

        # Minutes tag — rotation risk
        mins_tag = '⚠️' if mins_trend < -15 else ''

        rows.append({
            'player_id':     pid,
            'player':        web_name,
            'position':      pos,
            'team':          team_name,
            'price':         fpl_row['now_cost'] / 10,
            'status':        fpl_row['status'],
            'news':          fpl_row.get('news', ''),
            'ownership':     float(fpl_row.get('selected_by_percent', 0)),
            'form':          float(fpl_row.get('form', 0)),
            'opponent':      f"{'H' if fix1['is_home'] else 'A'} {opp_short}",
            'difficulty':    difficulty,
            'xP':            xp1,
            'xP_multi':      multi_xp_avg,
            'volatility':    vol,
            'captain_score': cap_score,
            'form_tag':      form_tag,
            'odds_tag':      odds_tag,
            'mins_tag':      mins_tag,
            'odds_mult':     odds_mult,
            'pts_avg_3':     round(float(pts_avg_3),  2) if not pd.isna(pts_avg_3)  else 0,
            'pts_avg_10':    round(float(pts_avg_10), 2) if not pd.isna(pts_avg_10) else 0,
            'next_5':        get_next_5(team_id),
            'total_points':  fpl_row.get('total_points', 0),
            'ppg':           float(fpl_row.get('points_per_game', 0)),
        })
    return pd.DataFrame(rows).sort_values('xP', ascending=False), next_gw

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
def optimize_squad(predictions_df, budget=100.0, existing_squad_ids=None,
                   free_transfers=1, use_multi_gw=False):
    df = predictions_df[predictions_df['status'] == 'a'].dropna(subset=['xP']).copy()
    df['pos_int'] = df['position'].map({'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})
    xp_col = 'xP_multi' if use_multi_gw and 'xP_multi' in df.columns else 'xP'
    df   = df.set_index('player_id')
    pids = df.index.tolist()

    prob     = pulp.LpProblem("FPL", pulp.LpMaximize)
    selected = {p: pulp.LpVariable(f"s_{p}", cat="Binary") for p in pids}
    starting = {p: pulp.LpVariable(f"x_{p}", cat="Binary") for p in pids}
    captain  = {p: pulp.LpVariable(f"c_{p}", cat="Binary") for p in pids}
    vice     = {p: pulp.LpVariable(f"v_{p}", cat="Binary") for p in pids}
    if existing_squad_ids:
        tin = {p: pulp.LpVariable(f"t_{p}", cat="Binary") for p in pids}

    base  = pulp.lpSum(starting[p] * df.loc[p, xp_col]          for p in pids)
    cap_b = pulp.lpSum(captain[p]  * df.loc[p, 'captain_score']  for p in pids)
    vc_b  = pulp.lpSum(vice[p]     * df.loc[p, 'captain_score'] * 0.5 for p in pids)

    if existing_squad_ids:
        n_tin   = pulp.lpSum(tin[p] for p in pids if p not in existing_squad_ids)
        prob   += base + cap_b + vc_b - 4 * (n_tin - free_transfers)
    else:
        prob += base + cap_b + vc_b

    prob += pulp.lpSum(selected.values()) == 15
    prob += pulp.lpSum(starting.values()) == 11
    prob += pulp.lpSum(captain.values())  == 1
    prob += pulp.lpSum(vice.values())     == 1

    for p in pids:
        prob += captain[p]  <= starting[p]
        prob += vice[p]     <= starting[p]
        prob += starting[p] <= selected[p]
        prob += captain[p] + vice[p] <= 1
        if df.loc[p, 'captain_score'] == 0:
            prob += captain[p] == 0
            prob += vice[p]    == 0

    for pos_int, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        pp = df[df['pos_int'] == pos_int].index.tolist()
        prob += pulp.lpSum(selected[p] for p in pp) == count

    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 1].index) == 1
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 2].index) >= 3
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 3].index) >= 2
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 4].index) >= 1
    prob += pulp.lpSum(selected[p] * df.loc[p, 'price'] for p in pids) <= budget

    for team in df['team'].unique():
        tp = df[df['team'] == team].index.tolist()
        prob += pulp.lpSum(selected[p] for p in tp) <= 3

    if existing_squad_ids:
        for p in pids:
            if p not in existing_squad_ids: prob += tin[p] >= selected[p]
            else:                           prob += tin[p] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None

    squad_ids   = [p for p in pids if selected[p].value() == 1]
    starter_ids = [p for p in pids if starting[p].value() == 1]
    cap_id      = next(p for p in pids if captain[p].value() == 1)
    vc_id       = next(p for p in pids if vice[p].value()    == 1)

    squad_df = df.loc[squad_ids].copy().reset_index()
    squad_df['is_starting']    = squad_df['player_id'].isin(starter_ids)
    squad_df['is_captain']     = squad_df['player_id'] == cap_id
    squad_df['is_vice']        = squad_df['player_id'] == vc_id
    squad_df['is_transfer_in'] = ~squad_df['player_id'].isin(existing_squad_ids) if existing_squad_ids else False
    squad_df['role'] = squad_df.apply(
        lambda r: '★ C' if r['is_captain'] else (
            'VC' if r['is_vice'] else ('START' if r['is_starting'] else 'BENCH')), axis=1)

    n_transfers = int(squad_df['is_transfer_in'].sum()) if existing_squad_ids else 0
    return {
        'squad':        squad_df.sort_values(['is_starting', 'pos_int'], ascending=[False, True]),
        'captain':      df.loc[cap_id, 'player'],
        'vice_captain': df.loc[vc_id, 'player'],
        'total_xP':     round(pulp.value(prob.objective), 2),
        'total_cost':   round(squad_df['price'].sum(), 1),
        'n_transfers':  n_transfers,
        'hits':         max(0, n_transfers - free_transfers) * 4,
    }

def optimize_lineup_from_squad(squad_pred):
    df = squad_pred.copy()
    df['pos_int'] = df['position'].map({'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})
    df   = df.set_index('player_id')
    pids = df.index.tolist()

    prob     = pulp.LpProblem("Lineup", pulp.LpMaximize)
    starting = {p: pulp.LpVariable(f"x_{p}", cat="Binary") for p in pids}
    captain  = {p: pulp.LpVariable(f"c_{p}", cat="Binary") for p in pids}
    vice     = {p: pulp.LpVariable(f"v_{p}", cat="Binary") for p in pids}

    prob += (pulp.lpSum(starting[p] * df.loc[p, 'xP']           for p in pids) +
             pulp.lpSum(captain[p]  * df.loc[p, 'captain_score'] for p in pids) +
             pulp.lpSum(vice[p]     * df.loc[p, 'captain_score'] * 0.5 for p in pids))

    prob += pulp.lpSum(starting.values()) == 11
    prob += pulp.lpSum(captain.values())  == 1
    prob += pulp.lpSum(vice.values())     == 1

    for p in pids:
        prob += captain[p] <= starting[p]
        prob += vice[p]    <= starting[p]
        prob += captain[p] + vice[p] <= 1
        if df.loc[p, 'captain_score'] == 0:
            prob += captain[p] == 0
            prob += vice[p]    == 0

    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 1].index) == 1
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 2].index) >= 3
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 3].index) >= 2
    prob += pulp.lpSum(starting[p] for p in df[df['pos_int'] == 4].index) >= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None

    starter_ids = [p for p in pids if starting[p].value() == 1]
    cap_id      = next(p for p in pids if captain[p].value() == 1)
    vc_id       = next(p for p in pids if vice[p].value()    == 1)

    df = df.reset_index()
    df['is_starting'] = df['player_id'].isin(starter_ids)
    df['is_captain']  = df['player_id'] == cap_id
    df['is_vice']     = df['player_id'] == vc_id
    df['role'] = df.apply(
        lambda r: '★ C' if r['is_captain'] else (
            'VC' if r['is_vice'] else ('START' if r['is_starting'] else 'BENCH')), axis=1)
    return df.sort_values(['is_starting', 'pos_int'], ascending=[False, True])

def analyse_weaknesses(my_pred, available):
    issues     = []
    league_avg = available.groupby('position')['xP'].mean().to_dict()
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        avg = league_avg.get(pos, 0)
        for _, p in my_pred[my_pred['position'] == pos].sort_values('xP').iterrows():
            if p['xP'] < avg * 0.75:
                issues.append({'player': p['player'], 'position': pos,
                                'xP': p['xP'], 'team_avg': round(avg, 2),
                                'gap': round(avg - p['xP'], 2),
                                'status': p['status'], 'news': p['news']})
    return sorted(issues, key=lambda x: x['gap'], reverse=True)

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:28px;letter-spacing:-0.5px;'>⚽ FPL xP Tool</h1>",
            unsafe_allow_html=True)

with st.spinner("Loading data..."):
    fpl_players, fpl_teams = fetch_bootstrap()
    fixtures = fetch_fixtures()

team_name_map = fpl_teams.set_index('id')['name'].to_dict()
fpl_players['team_name'] = fpl_players['team'].map(team_name_map)
fpl_players['position']  = fpl_players['element_type'].map(POS_MAP)
fpl_players['price']     = fpl_players['now_cost'] / 10
future  = fixtures[fixtures['finished'] == False]
next_gw = int(future['event'].dropna().min())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    team_id_input = st.text_input("FPL Team ID", placeholder="e.g. 1234567")
    bank, free_trs, team_info, my_squad_ids = None, None, None, None
    budget_input = 100.0

    if team_id_input.strip():
        try:
            tid       = int(team_id_input)
            r_info    = requests.get(f"{API}/entry/{tid}/", headers=HEADERS)
            team_info = r_info.json() if r_info.status_code == 200 else None
            r_picks   = requests.get(
                f"{API}/entry/{tid}/event/{max(1, next_gw - 1)}/picks/", headers=HEADERS)
            if r_picks.status_code == 200:
                picks_data   = r_picks.json()
                bank         = picks_data['entry_history']['bank'] / 10
                squad_value  = picks_data['entry_history']['value'] / 10
                used_last    = picks_data['entry_history']['event_transfers']
                free_trs     = 2 if used_last == 0 else 1
                my_squad_ids = pd.DataFrame(picks_data['picks'])['element'].tolist()
                budget_input = round(squad_value + bank, 1)
                st.success("✅ Team loaded")
                if team_info:
                    st.markdown(f"**{team_info.get('name', '')}**")
                    st.markdown(f"<span class='muted'>Rank</span> **{team_info.get('summary_overall_rank','?'):,}**", unsafe_allow_html=True)
                    st.markdown(f"<span class='muted'>Points</span> **{team_info.get('summary_overall_points','?')}**", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                c1.metric("Squad Value", f"£{round(squad_value,1)}m")
                c2.metric("Bank",        f"£{round(bank,1)}m")
                c1.metric("Budget",      f"£{budget_input}m")
                c2.metric("Free Xfers",  free_trs)
        except Exception as e:
            st.warning(f"Could not load team: {e}")

    free_transfers = free_trs if free_trs else st.selectbox("Free Transfers", [1, 2])
    if not team_id_input.strip():
        budget_input = st.number_input(
            "Budget (£m)", min_value=85.0, max_value=104.0, value=100.0, step=0.1)

    st.divider()
    st.markdown("**Odds API**")
    st.caption("Free key at the-odds-api.com")
    odds_api_key = st.text_input("API Key (optional)", type="password", placeholder="Paste key here")

    st.divider()
    st.markdown("**Chips Available**")
    st.caption("Tick what you still have")
    selected_chips = []
    if st.checkbox("Wildcard",       key="chip_wc"): selected_chips.append('wildcard')
    if st.checkbox("Bench Boost",    key="chip_bb"): selected_chips.append('bboost')
    if st.checkbox("Free Hit",       key="chip_fh"): selected_chips.append('freehit')
    if st.checkbox("Triple Captain", key="chip_tc"): selected_chips.append('3xc')

    st.divider()
    use_multi_gw = st.checkbox("Optimise for next 3 GWs", value=False)

    if st.button("🚀 Run Model", type="primary", use_container_width=True):
        for k in ['predictions', 'result', 'maes', 'next_gw', 'pipeline_key']:
            st.session_state.pop(k, None)
        st.session_state['model_run'] = True

if not st.session_state.get('model_run', False):
    st.markdown(f"<p style='color:#666'>Next GW: <b style='color:#fff'>GW{next_gw}</b> — Enter your Team ID and click Run Model</p>",
                unsafe_allow_html=True)
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────
eligible   = tuple(fpl_players[fpl_players['minutes'] > 90]['id'].tolist())
history_df = fetch_all_histories(eligible)

meta = fpl_players[['id', 'web_name', 'element_type', 'team', 'now_cost', 'position', 'team_name']].copy()
meta.columns = ['player_id', 'web_name', 'element_type', 'team', 'now_cost', 'position', 'team_name']
for col in ['total_points', 'minutes', 'expected_goals', 'expected_assists',
            'expected_goal_involvements', 'expected_goals_conceded',
            'goals_scored', 'assists', 'clean_sheets', 'bonus']:
    if col in history_df.columns:
        history_df[col] = pd.to_numeric(history_df[col], errors='coerce')
history_df = history_df.merge(meta, on='player_id', how='left')

# Odds
scorer_probs, cs_probs = {}, {}
if odds_api_key.strip():
    with st.spinner("Fetching market odds..."):
        scorer_probs, cs_probs = fetch_odds(odds_api_key.strip())
    if scorer_probs:
        st.sidebar.success(f"✅ Odds loaded — {len(scorer_probs)} players")
    else:
        st.sidebar.warning("⚠️ Odds unavailable — check key")

hist_features = None  # default in case cache is hit
cache_key = f"{next_gw}_{budget_input}_{use_multi_gw}_{bool(scorer_probs)}"
if st.session_state.get('pipeline_key') != cache_key:
    with st.spinner("Building features..."):
        features_df = build_features(history_df, fpl_teams)

    with st.spinner("Loading historical training data..."):
        hist_raw = fetch_historical_data()
        if not hist_raw.empty:
            hist_features = build_features(hist_raw, fpl_teams=None)
        else:
            hist_features = None
            st.sidebar.caption("⚠️ Historical data unavailable — training on current season only")

    with st.spinner("Training models..."):
        models, maes = train_position_models(features_df, hist_features)

    with st.spinner("Generating predictions..."):
        predictions, next_gw = predict_next_gw(
            features_df, models, fpl_players, fpl_teams, fixtures,
            scorer_probs, cs_probs, n_gws=3)

    with st.spinner("Optimizing squad..."):
        result = optimize_squad(predictions, budget=budget_input,
                                existing_squad_ids=my_squad_ids,
                                free_transfers=free_transfers,
                                use_multi_gw=use_multi_gw)
    st.session_state.update({
        'predictions':  predictions,
        'result':       result,
        'maes':         maes,
        'next_gw':      next_gw,
        'pipeline_key': cache_key,
    })
else:
    predictions = st.session_state['predictions']
    result      = st.session_state['result']
    maes        = st.session_state['maes']
    next_gw     = st.session_state['next_gw']

available = predictions[predictions['status'] == 'a'].copy()

# ── Header ────────────────────────────────────────────────────────────────────
if team_info:
    st.markdown(f"<h3>GW{next_gw} — {team_info.get('name','')}</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Rank",   f"{team_info.get('summary_overall_rank','?'):,}")
    c2.metric("Total Points",   team_info.get('summary_overall_points', '?'))
    c3.metric("Budget",         f"£{budget_input}m")
    c4.metric("Free Transfers", free_transfers)
else:
    st.markdown(f"<h3>GW{next_gw} Predictions</h3>", unsafe_allow_html=True)

st.caption(f"{'✅ Odds active' if scorer_probs else '➖ No odds'} | "
           f"{'✅ Historical data loaded' if hist_features is not None and not hist_features.empty else '➖ Current season only'}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE GKP", maes.get('GKP'))
c2.metric("MAE DEF", maes.get('DEF'))
c3.metric("MAE MID", maes.get('MID'))
c4.metric("MAE FWD", maes.get('FWD'))
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🔮 Predictions", "📋 My Lineup", "🔄 Transfers",
    "🏆 Optimal Squad", "🎯 Captain Picks", "📈 Differentials",
    "🗓️ Fixtures", "🃏 Chips"
])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"#### GW{next_gw} xP Predictions")
    st.caption("🔥 in form  ❄️ out of form  ➡️ consistent  |  📈 market boosted  📉 market dampened  |  ⚠️ minutes concern")
    c1, c2, c3 = st.columns([1, 1, 2])
    pos_filter  = c1.selectbox("Position", ["ALL", "GKP", "DEF", "MID", "FWD"], key="pred_pos")
    sort_filter = c2.selectbox("Sort by",  ["xP", "xP_multi", "captain_score", "price", "ownership"], key="pred_sort")
    search      = c3.text_input("Search player", "", key="pred_search")
    disp = available.copy()
    if pos_filter != "ALL": disp = disp[disp['position'] == pos_filter]
    if search: disp = disp[disp['player'].str.lower().str.contains(search.lower())]
    disp = disp.sort_values(sort_filter, ascending=False).head(100)
    st.dataframe(
        disp[['form_tag', 'odds_tag', 'mins_tag', 'player', 'position', 'team',
              'price', 'opponent', 'xP', 'xP_multi', 'captain_score',
              'pts_avg_3', 'pts_avg_10', 'ownership', 'ppg', 'next_5']].reset_index(drop=True),
        column_config={
            'form_tag':      st.column_config.TextColumn("Form"),
            'odds_tag':      st.column_config.TextColumn("Mkt"),
            'mins_tag':      st.column_config.TextColumn("Min"),
            'price':         st.column_config.NumberColumn("£",            format="£%.1f"),
            'xP':            st.column_config.NumberColumn("xP (next)",    format="%.2f"),
            'xP_multi':      st.column_config.NumberColumn("xP (3GW avg)", format="%.2f"),
            'captain_score': st.column_config.NumberColumn("Cap Score",    format="%.2f"),
            'pts_avg_3':     st.column_config.NumberColumn("Avg 3GW",      format="%.2f"),
            'pts_avg_10':    st.column_config.NumberColumn("Avg 10GW",     format="%.2f"),
            'ownership':     st.column_config.NumberColumn("Own%",         format="%.1f%%"),
            'next_5':        st.column_config.TextColumn("Next 5", width=300),
        },
        use_container_width=True, hide_index=True, height=600
    )

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Recommended Lineup From Your Squad")
    if not my_squad_ids:
        st.info("Enter your FPL Team ID to see your recommended lineup.")
    else:
        my_pred = predictions[predictions['player_id'].isin(my_squad_ids)].copy()
        lineup  = optimize_lineup_from_squad(my_pred)
        if lineup is not None:
            starters = lineup[lineup['is_starting']]
            bench    = lineup[~lineup['is_starting']]
            cap_row  = lineup[lineup['is_captain']]
            vc_row   = lineup[lineup['is_vice']]
            c1, c2, c3 = st.columns(3)
            c1.metric("Projected xP", round(starters['xP'].sum() + (cap_row.iloc[0]['xP'] if not cap_row.empty else 0), 2))
            c2.metric("Captain",      cap_row.iloc[0]['player'] if not cap_row.empty else '?')
            c3.metric("Vice Captain", vc_row.iloc[0]['player']  if not vc_row.empty else '?')
            st.markdown("**Starting XI**")
            st.dataframe(
                starters[['form_tag', 'odds_tag', 'mins_tag', 'player', 'position',
                           'team', 'price', 'opponent', 'xP', 'role', 'next_5']],
                column_config={
                    'form_tag': st.column_config.TextColumn("Form"),
                    'odds_tag': st.column_config.TextColumn("Mkt"),
                    'mins_tag': st.column_config.TextColumn("Min"),
                    'price':    st.column_config.NumberColumn("£",  format="£%.1f"),
                    'xP':       st.column_config.NumberColumn("xP", format="%.2f"),
                    'next_5':   st.column_config.TextColumn("Next 5", width=300),
                }, use_container_width=True, hide_index=True)
            st.markdown("**Bench**")
            st.dataframe(
                bench[['player', 'position', 'team', 'price', 'xP', 'role']],
                column_config={
                    'price': st.column_config.NumberColumn("£",  format="£%.1f"),
                    'xP':    st.column_config.NumberColumn("xP", format="%.2f"),
                }, use_container_width=True, hide_index=True)
        else:
            st.error("Could not generate lineup.")

        st.divider()
        st.markdown("#### 🔍 Squad Weaknesses")
        issues = analyse_weaknesses(my_pred, available)
        if not issues:
            st.success("No major weaknesses detected.")
        else:
            for issue in issues:
                gap_pct = round((issue['gap'] / max(issue['team_avg'], 0.01)) * 100)
                color   = "#ff4444" if gap_pct > 40 else "#f0a500"
                st.markdown(
                    f"<div style='background:#111;border:1px solid #222;border-radius:6px;"
                    f"padding:10px 16px;margin:4px 0;border-left:3px solid {color}'>"
                    f"<b>{issue['player']}</b> ({issue['position']}) — "
                    f"xP: <b>{issue['xP']}</b> vs avg <b>{issue['team_avg']}</b> "
                    f"<span style='color:{color}'>(-{issue['gap']} | {gap_pct}% below)</span>"
                    + (f"<br><span style='color:#888;font-size:12px'>⚠️ {issue['news']}</span>"
                       if issue['news'] else "") + "</div>", unsafe_allow_html=True)

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Transfer Planner")
    if not my_squad_ids:
        st.info("Enter your FPL Team ID to get transfer recommendations.")
    else:
        my_pred    = predictions[predictions['player_id'].isin(my_squad_ids)].copy()
        current_xp = my_pred['xP'].sum()
        st.markdown("**Your Current Squad**")
        st.dataframe(
            my_pred.sort_values(['position', 'xP'], ascending=[True, False])
                   [['form_tag', 'odds_tag', 'mins_tag', 'player', 'position', 'team',
                     'price', 'opponent', 'xP', 'xP_multi', 'captain_score',
                     'ownership', 'status', 'news']],
            column_config={
                'form_tag': st.column_config.TextColumn("Form"),
                'odds_tag': st.column_config.TextColumn("Mkt"),
                'mins_tag': st.column_config.TextColumn("Min"),
                'price':    st.column_config.NumberColumn("£",            format="£%.1f"),
                'xP':       st.column_config.NumberColumn("xP (next)",    format="%.2f"),
                'xP_multi': st.column_config.NumberColumn("xP (3GW avg)", format="%.2f"),
            }, use_container_width=True, hide_index=True)
        st.metric("Current Squad xP", round(current_xp, 2))
        st.divider()

        st.markdown(f"**Recommended Transfers** — Bank: £{round(bank or 0, 1)}m | Free: {free_transfers}")
        with st.spinner("Calculating best transfers..."):
            scenarios, seen = [], set()
            for n in range(1, 4):
                res = optimize_squad(predictions, budget=budget_input,
                                     existing_squad_ids=my_squad_ids,
                                     free_transfers=free_transfers,
                                     use_multi_gw=use_multi_gw)
                if res is None: continue
                new_ids = res['squad']['player_id'].tolist()
                out_ids = [p for p in my_squad_ids if p not in new_ids]
                in_ids  = [p for p in new_ids       if p not in my_squad_ids]
                if not in_ids: continue
                key = (frozenset(out_ids), frozenset(in_ids))
                if key in seen: continue
                seen.add(key)
                actual_hit = max(0, len(in_ids) - free_transfers) * 4
                net_gain   = round(res['total_xP'] - current_xp - actual_hit, 2)
                scenarios.append({'n': len(in_ids), 'hit': actual_hit, 'net_gain': net_gain,
                                   'total_xP': res['total_xP'], 'result': res,
                                   'out_ids': out_ids, 'in_ids': in_ids})

        if not scenarios:
            st.success("Your squad is already optimal — no beneficial transfers found.")
        else:
            for s in sorted(scenarios, key=lambda x: x['net_gain'], reverse=True):
                hit_str  = f"-{s['hit']}pt hit" if s['hit'] > 0 else "Free"
                gain_str = f"+{s['net_gain']}" if s['net_gain'] > 0 else str(s['net_gain'])
                with st.expander(
                    f"{s['n']} Transfer{'s' if s['n'] > 1 else ''} | {hit_str} | Net xP: {gain_str}",
                    expanded=(s['n'] == 1)):
                    out_players = my_pred[my_pred['player_id'].isin(s['out_ids'])]
                    in_players  = s['result']['squad'][s['result']['squad']['player_id'].isin(s['in_ids'])]
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("🔴 **SELL**")
                        for _, r in out_players.iterrows():
                            st.markdown(
                                f"<div style='background:#1a0a0a;border:1px solid #331111;"
                                f"border-radius:4px;padding:8px 12px;margin:4px 0'>"
                                f"{r.get('form_tag','➡️')} {r.get('odds_tag','➖')} {r.get('mins_tag','')}"
                                f" <b>{r['player']}</b> £{r['price']} | xP <b>{r['xP']}</b> | {r['opponent']}"
                                f"</div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown("🟢 **BUY**")
                        for _, r in in_players.iterrows():
                            st.markdown(
                                f"<div style='background:#0a1a0a;border:1px solid #113311;"
                                f"border-radius:4px;padding:8px 12px;margin:4px 0'>"
                                f"{r.get('form_tag','➡️')} {r.get('odds_tag','➖')} {r.get('mins_tag','')}"
                                f" <b>{r['player']}</b> £{r['price']} | xP <b>{r['xP']}</b> | "
                                f"3GW <b>{r.get('xP_multi','?')}</b> | {r['opponent']}"
                                f"</div>", unsafe_allow_html=True)
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("New Squad xP", s['total_xP'])
                    col2.metric("Points Hit",   f"-{s['hit']}" if s['hit'] > 0 else "None")
                    col3.metric("Net xP Gain",  gain_str)

# ── Tab 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Optimal Squad")
    if result:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total xP",     result['total_xP'])
        c2.metric("Cost",         f"£{result['total_cost']}m")
        c3.metric("Captain",      result['captain'])
        c4.metric("Vice Captain", result['vice_captain'])
        if result['n_transfers'] > 0:
            st.warning(f"⚠️ {result['n_transfers']} transfers needed — -{result['hits']}pt hit")
        sq = result['squad']
        st.markdown("**Starting XI**")
        st.dataframe(
            sq[sq['is_starting']][['form_tag', 'odds_tag', 'mins_tag', 'player', 'position',
                                   'team', 'price', 'opponent', 'xP', 'xP_multi',
                                   'role', 'is_transfer_in', 'next_5']],
            column_config={
                'form_tag':       st.column_config.TextColumn("Form"),
                'odds_tag':       st.column_config.TextColumn("Mkt"),
                'mins_tag':       st.column_config.TextColumn("Min"),
                'price':          st.column_config.NumberColumn("£",            format="£%.1f"),
                'xP':             st.column_config.NumberColumn("xP (next)",    format="%.2f"),
                'xP_multi':       st.column_config.NumberColumn("xP (3GW avg)", format="%.2f"),
                'is_transfer_in': st.column_config.CheckboxColumn("Transfer In"),
                'next_5':         st.column_config.TextColumn("Next 5", width=300),
            }, use_container_width=True, hide_index=True)
        st.markdown("**Bench**")
        st.dataframe(
            sq[~sq['is_starting']][['player', 'position', 'team', 'price', 'xP', 'is_transfer_in']],
            column_config={
                'price':          st.column_config.NumberColumn("£",  format="£%.1f"),
                'xP':             st.column_config.NumberColumn("xP", format="%.2f"),
                'is_transfer_in': st.column_config.CheckboxColumn("Transfer In"),
            }, use_container_width=True, hide_index=True)
    else:
        st.error("Optimizer could not find a valid squad.")

# ── Tab 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown(f"#### Captain Picks — GW{next_gw}")
    st.caption("Score = (xP + 0.3 × vol) × fixture difficulty multiplier. xP adjusted for form, odds, home/away bias.")
    cap_df = available[available['captain_score'] > 0].sort_values('captain_score', ascending=False).head(15)
    for i, (_, r) in enumerate(cap_df.iterrows()):
        medal      = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        bg_color   = "#1a1a0a" if i == 0 else "#111"
        diff_color = "#00cc66" if r['difficulty'] <= 2 else "#f0a500" if r['difficulty'] == 3 else "#ff4444"
        st.markdown(
            f"<div style='background:{bg_color};border:1px solid #222;border-radius:6px;"
            f"padding:10px 16px;margin:4px 0'>"
            f"<span style='font-size:18px'>{medal}</span> "
            f"{r.get('form_tag','➡️')} {r.get('odds_tag','➖')} "
            f"<b>{r['player']}</b> <span style='color:#666'>— {r['team']}</span> | "
            f"xP: <b>{r['xP']}</b> | 3GW: {r.get('xP_multi','?')} | "
            f"vs {r['opponent']} <span style='color:{diff_color}'>[diff {r['difficulty']}]</span> | "
            f"Cap Score: <b>{r['captain_score']}</b>"
            f"</div>", unsafe_allow_html=True)

# ── Tab 6 ─────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("#### Differentials & Value Picks")
    c1, c2 = st.columns(2)
    own_threshold = c1.slider("Max ownership %", 5, 30, 15, key="diff_own")
    min_xp        = c2.slider("Min xP", 2.0, 8.0, 3.5, step=0.5, key="diff_xp")
    diffs = available[
        (available['ownership'] <= own_threshold) & (available['xP'] >= min_xp)
    ].sort_values('captain_score', ascending=False)
    st.caption(f"{len(diffs)} differentials found")
    for pos in ['FWD', 'MID', 'DEF', 'GKP']:
        pos_diffs = diffs[diffs['position'] == pos].head(8)
        if pos_diffs.empty: continue
        st.markdown(f"**{pos}**")
        st.dataframe(
            pos_diffs[['form_tag', 'odds_tag', 'mins_tag', 'player', 'team', 'price',
                       'opponent', 'xP', 'xP_multi', 'captain_score',
                       'ownership', 'next_5']].reset_index(drop=True),
            column_config={
                'form_tag':      st.column_config.TextColumn("Form"),
                'odds_tag':      st.column_config.TextColumn("Mkt"),
                'mins_tag':      st.column_config.TextColumn("Min"),
                'price':         st.column_config.NumberColumn("£",            format="£%.1f"),
                'xP':            st.column_config.NumberColumn("xP (next)",    format="%.2f"),
                'xP_multi':      st.column_config.NumberColumn("xP (3GW avg)", format="%.2f"),
                'captain_score': st.column_config.NumberColumn("Cap Score",    format="%.2f"),
                'ownership':     st.column_config.NumberColumn("Own%",         format="%.1f%%"),
                'next_5':        st.column_config.TextColumn("Next 5", width=300),
            }, use_container_width=True, hide_index=True)

# ── Tab 7 ─────────────────────────────────────────────────────────────────────
with tab7:
    st.markdown("#### Fixture Difficulty Planner")
    n_gws     = st.slider("GWs ahead", 3, 8, 5, key="fix_gws")
    future_n  = fixtures[fixtures['finished'] == False].copy()
    gws_ahead = sorted(future_n['event'].dropna().unique())[:n_gws]
    ts        = fpl_teams.set_index('id')
    fix_rows  = []
    for _, team in fpl_teams.iterrows():
        row = {'Team': team['name']}
        total_diff = 0
        for gw in gws_ahead:
            home = future_n[(future_n['team_h'] == team['id']) & (future_n['event'] == gw)]
            away = future_n[(future_n['team_a'] == team['id']) & (future_n['event'] == gw)]
            if not home.empty:
                r    = home.iloc[0]
                opp  = ts.loc[r['team_a'], 'short_name'] if r['team_a'] in ts.index else '?'
                diff = int(r['team_h_difficulty'])
                row[f'GW{int(gw)}'] = f"{opp}(H)[{diff}]"; total_diff += diff
            elif not away.empty:
                r    = away.iloc[0]
                opp  = ts.loc[r['team_h'], 'short_name'] if r['team_h'] in ts.index else '?'
                diff = int(r['team_a_difficulty'])
                row[f'GW{int(gw)}'] = f"{opp}(A)[{diff}]"; total_diff += diff
            else:
                row[f'GW{int(gw)}'] = "BGW"; total_diff += 3
        row['Avg Diff'] = round(total_diff / n_gws, 1)
        fix_rows.append(row)
    fix_df  = pd.DataFrame(fix_rows).sort_values('Avg Diff')
    gw_cols = [f'GW{int(g)}' for g in gws_ahead]
    def color_diff(val):
        try:
            d = int(str(val).split('[')[1].replace(']', ''))
            if d <= 2: return 'color: #00cc66; font-weight: bold'
            if d == 3: return 'color: #f0a500'
            return 'color: #ff4444'
        except: return 'color: #666'
    st.dataframe(fix_df.style.map(color_diff, subset=gw_cols),
                 use_container_width=True, hide_index=True, height=600)

# ── Tab 8 ─────────────────────────────────────────────────────────────────────
with tab8:
    st.markdown("#### Chip Strategy")
    if not selected_chips:
        st.info("Tick the chips you still have in the sidebar to see recommendations.")
    else:
        future_n  = fixtures[fixtures['finished'] == False].copy()
        gw_scores = []
        for gw in sorted(future_n['event'].dropna().unique())[:10]:
            gw_fix = future_n[future_n['event'] == gw]
            if not gw_fix.empty:
                gw_scores.append({'gw': int(gw),
                                   'avg_diff': round(gw_fix[['team_h_difficulty','team_a_difficulty']].mean().mean(), 2)})
        gw_df = pd.DataFrame(gw_scores).sort_values('avg_diff') if gw_scores else pd.DataFrame()

        if 'wildcard' in selected_chips:
            st.markdown("---")
            st.markdown("### 🃏 Wildcard")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                **Use when:**
                - 4+ players you want to replace
                - Major injury crisis across multiple positions
                - Run of great fixtures for teams you don't own
                - After a double GW announcement
                """)
                if not gw_df.empty:
                    best = gw_df.iloc[0]
                    st.markdown(f"**Best upcoming GW:** GW{int(best['gw'])} *(avg diff {best['avg_diff']})*")
            with c2:
                if my_squad_ids:
                    st.markdown("**Weakest to replace:**")
                    for _, r in predictions[predictions['player_id'].isin(my_squad_ids)].sort_values('xP').head(5).iterrows():
                        st.markdown(
                            f"<div style='background:#111;border:1px solid #222;border-radius:4px;"
                            f"padding:8px 12px;margin:3px 0'>"
                            f"{r.get('form_tag','➡️')} <b>{r['player']}</b> ({r['position']}) — "
                            f"xP: <span style='color:#ff4444'>{r['xP']}</span></div>",
                            unsafe_allow_html=True)

        if 'bboost' in selected_chips:
            st.markdown("---")
            st.markdown("### 🚀 Bench Boost")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                **Use when:**
                - Double GW with 3-4 bench players who have doubles
                - Never use in a blank gameweek
                """)
                if not gw_df.empty:
                    best = gw_df.iloc[0]
                    st.markdown(f"**Best upcoming GW:** GW{int(best['gw'])} *(avg diff {best['avg_diff']})*")
            with c2:
                if result:
                    bench_players = result['squad'][~result['squad']['is_starting']]
                    bench_xp      = bench_players['xP'].sum()
                    color         = "#00cc66" if bench_xp >= 12 else "#ff4444"
                    st.markdown(f"**Bench xP:** <span style='color:{color}'>{round(bench_xp,2)}</span>",
                                unsafe_allow_html=True)
                    for _, r in bench_players.iterrows():
                        xp_c = "#00cc66" if r['xP'] > 3 else "#888"
                        st.markdown(
                            f"<div style='background:#111;border:1px solid #222;border-radius:4px;"
                            f"padding:8px 12px;margin:3px 0'>"
                            f"<b>{r['player']}</b> ({r['position']}) — "
                            f"xP: <span style='color:{xp_c}'>{r['xP']}</span> | {r['opponent']}</div>",
                            unsafe_allow_html=True)
                    if bench_xp < 12:
                        st.warning("⚠️ Bench xP is low. Upgrade bench before using this.")

        if 'freehit' in selected_chips:
            st.markdown("---")
            st.markdown("### 🎯 Free Hit")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                **Use when:**
                - Blank gameweek — field a full 11 playing players
                - Squad resets after — never waste on a normal GW
                """)
                if not gw_df.empty:
                    best = gw_df.iloc[0]
                    st.markdown(f"**Best upcoming GW:** GW{int(best['gw'])} *(avg diff {best['avg_diff']})*")
            with c2:
                st.markdown(f"**Optimal Free Hit squad GW{next_gw}:**")
                fh_result = optimize_squad(predictions, budget=104.0,
                                           existing_squad_ids=None, free_transfers=15)
                if fh_result:
                    sq = fh_result['squad']
                    c3, c4, c5 = st.columns(3)
                    c3.metric("Total xP", fh_result['total_xP'])
                    c4.metric("Cost",     f"£{fh_result['total_cost']}m")
                    c5.metric("Captain",  fh_result['captain'])
                    st.dataframe(
                        sq[sq['is_starting']][['player', 'position', 'team', 'price', 'opponent', 'xP', 'role']],
                        column_config={
                            'price': st.column_config.NumberColumn("£",  format="£%.1f"),
                            'xP':    st.column_config.NumberColumn("xP", format="%.2f"),
                        }, use_container_width=True, hide_index=True)

        if '3xc' in selected_chips:
            st.markdown("---")
            st.markdown("### ⚡ Triple Captain")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                **Use when:**
                - Your captain has a double gameweek
                - Exceptional form + easy fixture (diff 1-2)
                - Never on a difficulty 4-5 fixture
                """)
            with c2:
                st.markdown(f"**Best candidates GW{next_gw}:**")
                for i, (_, r) in enumerate(
                    available[available['captain_score'] > 0].sort_values(
                        'captain_score', ascending=False).head(5).iterrows()):
                    diff_color = "#00cc66" if r['difficulty'] <= 2 else "#f0a500" if r['difficulty'] == 3 else "#ff4444"
                    rank_color = "#f0a500" if i == 0 else "#888"
                    st.markdown(
                        f"<div style='background:#111;border:1px solid #222;border-radius:4px;"
                        f"padding:8px 12px;margin:3px 0'>"
                        f"<span style='color:{rank_color};font-weight:700'>{i+1}.</span> "
                        f"{r.get('form_tag','➡️')} {r.get('odds_tag','➖')} "
                        f"<b>{r['player']}</b> ({r['team']}) — "
                        f"xP: <b>{r['xP']}</b> | Cap: <b>{r['captain_score']}</b> | "
                        f"vs {r['opponent']} <span style='color:{diff_color}'>[diff {r['difficulty']}]</span>"
                        f"</div>", unsafe_allow_html=True)