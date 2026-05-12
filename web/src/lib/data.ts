import { supabase } from "./supabase";
import type { PlayerStat, PlayerView, Prediction } from "../types";

function toNumber(value: unknown, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export async function fetchPredictions(): Promise<Prediction[]> {
  if (!supabase) return [];

  const { data, error } = await supabase
    .from("predictions")
    .select("*")
    .order("predicted_points", { ascending: false });

  if (error) throw error;
  return (data ?? []).map((row) => ({
    ...row,
    player_id: toNumber(row.player_id),
    gameweek: toNumber(row.gameweek),
    predicted_points: toNumber(row.predicted_points),
    now_cost: toNumber(row.now_cost),
  })) as Prediction[];
}

export async function fetchPlayerStats(): Promise<PlayerStat[]> {
  if (!supabase) return [];

  const { data, error } = await supabase
    .from("player_gameweek_stats")
    .select("*");

  if (error) throw error;
  return (data ?? []).map((row) => ({
    ...row,
    player_id: toNumber(row.player_id),
    now_cost: toNumber(row.now_cost),
    selected_by_percent: toNumber(row.selected_by_percent),
    total_points: toNumber(row.total_points),
    form: toNumber(row.form),
    ict_index: toNumber(row.ict_index),
    influence: toNumber(row.influence),
    creativity: toNumber(row.creativity),
    threat: toNumber(row.threat),
    transfers_in: toNumber(row.transfers_in),
    transfers_out: toNumber(row.transfers_out),
    minutes: toNumber(row.minutes),
    points_per_game: toNumber(row.points_per_game),
    gameweek: toNumber(row.gameweek),
  })) as PlayerStat[];
}

export function buildPlayerViews(predictions: Prediction[], stats: PlayerStat[]): PlayerView[] {
  const statById = new Map(stats.map((stat) => [stat.player_id, stat]));

  return predictions.map((prediction) => {
    const stat = statById.get(prediction.player_id);
    const price = prediction.now_cost / 10;
    return {
      ...prediction,
      price,
      valueScore: price > 0 ? prediction.predicted_points / price : 0,
      selectedByPercent: stat?.selected_by_percent,
      form: stat?.form,
      ictIndex: stat?.ict_index,
      minutes: stat?.minutes,
      netTransfers: stat ? stat.transfers_in - stat.transfers_out : undefined,
    };
  });
}
