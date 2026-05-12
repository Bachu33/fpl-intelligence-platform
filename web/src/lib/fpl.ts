import type { PlayerView, Position } from "../types";

export const positions: Position[] = ["GKP", "DEF", "MID", "FWD"];

export const positionColors: Record<Position, string> = {
  GKP: "#ebff00",
  DEF: "#00ff87",
  MID: "#05f0ff",
  FWD: "#ff4c4c",
};

export function formatPrice(price: number) {
  return `£${price.toFixed(1)}m`;
}

export function topByPosition(players: PlayerView[], position: Position) {
  return players
    .filter((player) => player.position === position)
    .sort((a, b) => b.predicted_points - a.predicted_points)[0];
}

export function minutesRisk(player: PlayerView) {
  const minutes = player.minutes ?? 0;
  if (minutes >= 2200) return "Nailed";
  if (minutes >= 1000) return "Managed";
  return "Minutes risk";
}

export function pickSquad(players: PlayerView[], budget: number, maxPerTeam: number) {
  const targets: Record<Position, number> = { GKP: 2, DEF: 5, MID: 5, FWD: 3 };
  const selected: PlayerView[] = [];
  const teamCounts = new Map<string, number>();
  let spent = 0;

  for (const position of positions) {
    const candidates = players
      .filter((player) => player.position === position)
      .sort((a, b) => b.predicted_points - a.predicted_points);

    for (const player of candidates) {
      if (selected.filter((pick) => pick.position === position).length >= targets[position]) break;
      if ((teamCounts.get(player.team) ?? 0) >= maxPerTeam) continue;
      if (spent + player.price > budget) continue;
      selected.push(player);
      spent += player.price;
      teamCounts.set(player.team, (teamCounts.get(player.team) ?? 0) + 1);
    }
  }

  return selected;
}
