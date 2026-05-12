export type Position = "GKP" | "DEF" | "MID" | "FWD";

export type Prediction = {
  id?: number;
  player_id: number;
  player_name: string;
  team: string;
  position: Position;
  gameweek: number;
  season: string;
  predicted_points: number;
  now_cost: number;
  updated_at?: string;
};

export type PlayerStat = {
  player_id: number;
  player_name: string;
  team: string;
  position: Position;
  now_cost: number;
  selected_by_percent: number;
  total_points: number;
  form: number;
  ict_index: number;
  influence: number;
  creativity: number;
  threat: number;
  transfers_in: number;
  transfers_out: number;
  minutes: number;
  points_per_game: number;
  gameweek: number;
};

export type PlayerView = Prediction & {
  price: number;
  valueScore: number;
  selectedByPercent?: number;
  form?: number;
  ictIndex?: number;
  minutes?: number;
  netTransfers?: number;
};
