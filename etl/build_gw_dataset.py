import json
from datetime import datetime
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
SUMMARY_DIR = RAW_DIR / "element_summaries"
PROCESSED_DIR = Path("data/processed")

PLAYER_GW_PATH = PROCESSED_DIR / "player_gameweek_features.parquet"
PREDICTION_FEATURES_PATH = PROCESSED_DIR / "prediction_features.parquet"

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
SEASON = "2024-25"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_next_gameweek(events):
    for event in events:
        if event.get("is_next"):
            return event["id"]
    for event in events:
        if event.get("is_current"):
            return event["id"] + 1
    finished = [event["id"] for event in events if event.get("finished")]
    return max(finished) + 1 if finished else 1


def fixture_lookup(fixtures):
    rows = []

    for fixture in fixtures:
        event = fixture.get("event")
        if event is None:
            continue

        rows.append({
            "team_id": fixture["team_h"],
            "round": event,
            "opponent_team_id": fixture["team_a"],
            "was_home": 1,
            "fdr": fixture.get("team_h_difficulty"),
            "fixture_count": 1,
        })
        rows.append({
            "team_id": fixture["team_a"],
            "round": event,
            "opponent_team_id": fixture["team_h"],
            "was_home": 0,
            "fdr": fixture.get("team_a_difficulty"),
            "fixture_count": 1,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return (
        df.groupby(["team_id", "round"], as_index=False)
        .agg({
            "opponent_team_id": "first",
            "was_home": "mean",
            "fdr": "mean",
            "fixture_count": "sum",
        })
    )


def load_player_metadata(bootstrap):
    teams = {team["id"]: team["name"] for team in bootstrap["teams"]}

    records = []
    for player in bootstrap["elements"]:
        records.append({
            "player_id": player["id"],
            "player_name": player["web_name"],
            "team_id": player["team"],
            "team": teams.get(player["team"], "Unknown"),
            "position": POSITION_MAP.get(player["element_type"], "Unknown"),
            "now_cost": player["now_cost"],
            "season": SEASON,
        })

    return pd.DataFrame(records), teams


def load_history_rows(metadata):
    rows = []

    for _, player in metadata.iterrows():
        path = SUMMARY_DIR / f"{player['player_id']}.json"
        if not path.exists():
            print(f"WARNING: Missing element summary for player {player['player_id']}")
            continue

        summary = load_json(path)
        for history in summary.get("history", []):
            rows.append({
                "player_id": player["player_id"],
                "player_name": player["player_name"],
                "team_id": player["team_id"],
                "team": player["team"],
                "position": player["position"],
                "season": player["season"],
                "round": history.get("round"),
                "total_points": history.get("total_points", 0),
                "minutes": history.get("minutes", 0),
                "starts": history.get("starts", 0),
                "goals_scored": history.get("goals_scored", 0),
                "assists": history.get("assists", 0),
                "clean_sheets": history.get("clean_sheets", 0),
                "goals_conceded": history.get("goals_conceded", 0),
                "saves": history.get("saves", 0),
                "bonus": history.get("bonus", 0),
                "bps": history.get("bps", 0),
                "influence": history.get("influence", 0),
                "creativity": history.get("creativity", 0),
                "threat": history.get("threat", 0),
                "ict_index": history.get("ict_index", 0),
                "expected_goals": history.get("expected_goals", 0),
                "expected_assists": history.get("expected_assists", 0),
                "expected_goal_involvements": history.get("expected_goal_involvements", 0),
                "expected_goals_conceded": history.get("expected_goals_conceded", 0),
                "value": history.get("value", player["now_cost"]),
                "selected": history.get("selected", 0),
                "transfers_balance": history.get("transfers_balance", 0),
                "transfers_in": history.get("transfers_in", 0),
                "transfers_out": history.get("transfers_out", 0),
            })

    return pd.DataFrame(rows)


def aggregate_gameweeks(history):
    numeric_sum_cols = [
        "total_points", "minutes", "starts", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "transfers_balance", "transfers_in",
        "transfers_out",
    ]

    numeric_cols = numeric_sum_cols + ["value", "selected"]
    for col in numeric_cols:
        history[col] = pd.to_numeric(history[col], errors="coerce").fillna(0)

    agg_map = {col: "sum" for col in numeric_sum_cols}
    agg_map.update({
        "player_name": "first",
        "team_id": "first",
        "team": "first",
        "position": "first",
        "season": "first",
        "value": "last",
        "selected": "last",
    })

    return (
        history.groupby(["player_id", "round"], as_index=False)
        .agg(agg_map)
        .sort_values(["player_id", "round"])
        .reset_index(drop=True)
    )


def add_lagged_features(df):
    lag_cols = [
        "total_points", "minutes", "starts", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "value", "selected",
        "transfers_balance", "transfers_in", "transfers_out",
    ]

    df = df.sort_values(["player_id", "round"]).copy()
    grouped = df.groupby("player_id", group_keys=False)

    for col in lag_cols:
        df[f"{col}_lag1"] = grouped[col].shift(1)

    rolling_cols = [
        "total_points", "minutes", "starts", "ict_index", "influence",
        "creativity", "threat", "expected_goal_involvements",
        "expected_goals_conceded", "bps",
    ]

    for col in rolling_cols:
        shifted = grouped[col].shift(1)
        df[f"{col}_roll3"] = (
            shifted.groupby(df["player_id"])
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"{col}_roll5"] = (
            shifted.groupby(df["player_id"])
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    df["games_played_before"] = grouped.cumcount()
    df["target_points"] = df["total_points"]
    return df


def build_training_frame(history, fixtures):
    df = aggregate_gameweeks(history)
    df = df.merge(fixtures, how="left", left_on=["team_id", "round"], right_on=["team_id", "round"])
    df = add_lagged_features(df)

    df["fdr"] = df["fdr"].fillna(3)
    df["fixture_count"] = df["fixture_count"].fillna(0)
    df["was_home"] = df["was_home"].fillna(0.5)
    df["opponent_team_id"] = df["opponent_team_id"].fillna(0).astype(int)

    feature_cols = [col for col in df.columns if col.endswith("_lag1") or col.endswith("_roll3") or col.endswith("_roll5")]
    df[feature_cols] = df[feature_cols].fillna(0)

    df = df[df["games_played_before"] > 0].copy()
    return df


def build_prediction_frame(training_frame, metadata, fixtures, next_gw):
    latest = (
        training_frame.sort_values(["player_id", "round"])
        .groupby("player_id", as_index=False)
        .tail(1)
    )

    base_cols = [
        "player_id", "player_name", "team_id", "team", "position", "season",
        "value", "selected", "transfers_balance", "transfers_in", "transfers_out",
        "games_played_before",
    ]
    prediction = latest[base_cols].copy()
    prediction["round"] = next_gw
    prediction["gameweek"] = next_gw
    prediction["now_cost"] = prediction["value"]

    upcoming = fixtures[fixtures["round"] == next_gw]
    prediction = prediction.merge(
        upcoming,
        how="left",
        left_on=["team_id", "round"],
        right_on=["team_id", "round"],
    )

    prediction["fdr"] = prediction["fdr"].fillna(5)
    prediction["fixture_count"] = prediction["fixture_count"].fillna(0)
    prediction["was_home"] = prediction["was_home"].fillna(0.5)
    prediction["opponent_team_id"] = prediction["opponent_team_id"].fillna(0).astype(int)

    actual_columns = [
        "total_points", "minutes", "starts", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "value", "selected",
        "transfers_balance", "transfers_in", "transfers_out",
    ]

    latest_by_player = latest.set_index("player_id")
    history_by_player = training_frame.sort_values(["player_id", "round"]).groupby("player_id")

    for col in actual_columns:
        prediction[f"{col}_lag1"] = prediction["player_id"].map(latest_by_player[col]).fillna(0)

    rolling_columns = [
        "total_points", "minutes", "starts", "ict_index", "influence",
        "creativity", "threat", "expected_goal_involvements",
        "expected_goals_conceded", "bps",
    ]

    rolling_maps = {}
    for col in rolling_columns:
        rolling_maps[f"{col}_roll3"] = history_by_player[col].apply(lambda s: s.tail(3).mean()).to_dict()
        rolling_maps[f"{col}_roll5"] = history_by_player[col].apply(lambda s: s.tail(5).mean()).to_dict()

    for col, values in rolling_maps.items():
        prediction[col] = prediction["player_id"].map(values).fillna(0)

    prediction["games_played_before"] = prediction["games_played_before"] + 1
    prediction["now_cost"] = prediction["now_cost"].fillna(
        prediction["player_id"].map(metadata.set_index("player_id")["now_cost"])
    )

    return prediction


if __name__ == "__main__":
    print("Building player-gameweek training dataset...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    bootstrap = load_json(RAW_DIR / "bootstrap.json")
    fixtures_raw = load_json(RAW_DIR / "fixtures.json")

    metadata, teams = load_player_metadata(bootstrap)
    history = load_history_rows(metadata)

    if history.empty:
        print("FATAL: No player history found. Run etl/fetch_data.py first.")
        raise SystemExit(1)

    fixtures = fixture_lookup(fixtures_raw)
    training = build_training_frame(history, fixtures)
    next_gw = get_next_gameweek(bootstrap["events"])
    prediction = build_prediction_frame(training, metadata, fixtures, next_gw)

    training.to_parquet(PLAYER_GW_PATH, index=False)
    prediction.to_parquet(PREDICTION_FEATURES_PATH, index=False)

    print(f"[{datetime.now()}] Saved {len(training)} training rows to {PLAYER_GW_PATH}")
    print(f"[{datetime.now()}] Saved {len(prediction)} prediction rows for GW{next_gw} to {PREDICTION_FEATURES_PATH}")
