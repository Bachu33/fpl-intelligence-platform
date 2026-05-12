import type { ReactNode } from "react";
import type { PlayerView } from "../types";
import { formatPrice, minutesRisk, positionColors } from "../lib/fpl";

export function Kicker({ children }: { children: ReactNode }) {
  return <div className="kicker">{children}</div>;
}

export function PageHeader({ kicker, title, subtitle, badges = [] }: {
  kicker: string;
  title: string;
  subtitle: string;
  badges?: string[];
}) {
  return (
    <header className="page-header">
      <div>
        <Kicker>{kicker}</Kicker>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <div className="badge-row">
        {badges.map((badge) => <span className="badge" key={badge}>{badge}</span>)}
      </div>
    </header>
  );
}

export function StatCard({ label, value, sub }: { label: string; value: ReactNode; sub?: string }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub ? <div className="stat-sub">{sub}</div> : null}
    </div>
  );
}

export function PlayerCard({ player, rank, captain = false }: { player: PlayerView; rank?: string | number; captain?: boolean }) {
  const color = positionColors[player.position];
  return (
    <article className="player-card" style={{ borderTopColor: color }}>
      <div className="card-topline">
        <span className="position-badge" style={{ backgroundColor: color }}>{rank ?? player.position}</span>
        <span className="badge outline">{player.position}</span>
        <span className="badge outline">{minutesRisk(player)}</span>
      </div>
      <h3>{player.player_name}</h3>
      <p>{player.team} · {formatPrice(player.price)}</p>
      <div className="xpts">{player.predicted_points.toFixed(2)}</div>
      <div className="stat-sub">{captain ? `Captain ${ (player.predicted_points * 2).toFixed(2) } pts` : "predicted points"}</div>
    </article>
  );
}

export function DataTable({ children }: { children: ReactNode }) {
  return <div className="table-card">{children}</div>;
}
