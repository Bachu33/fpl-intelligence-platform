import { useEffect, useMemo, useState } from "react";
import { Shell } from "./components/Shell";
import { DataTable, PageHeader, PlayerCard, StatCard } from "./components/Ui";
import { buildPlayerViews, fetchPlayerStats, fetchPredictions } from "./lib/data";
import { formatPrice, pickSquad, positions, topByPosition } from "./lib/fpl";
import { hasSupabaseConfig } from "./lib/supabase";
import type { PlayerStat, PlayerView, Position, Prediction } from "./types";

type Page = "dashboard" | "picks" | "captain" | "fixtures" | "prices" | "optimizer" | "team";

function useFplData() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [stats, setStats] = useState<PlayerStat[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        const [predictionRows, statRows] = await Promise.all([fetchPredictions(), fetchPlayerStats()]);
        setPredictions(predictionRows);
        setStats(statRows);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load Supabase data");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return { predictions, stats, loading, error };
}

function PlayerTable({ players, captain = false }: { players: PlayerView[]; captain?: boolean }) {
  return (
    <DataTable>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Player</th>
            <th>Team</th>
            <th>Pos</th>
            <th>Price</th>
            <th>xPts</th>
            {captain ? <th>Captain</th> : null}
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {players.map((player, index) => (
            <tr key={player.player_id}>
              <td>{index + 1}</td>
              <td>{player.player_name}</td>
              <td>{player.team}</td>
              <td>{player.position}</td>
              <td>{formatPrice(player.price)}</td>
              <td><Progress value={player.predicted_points} max={15} /></td>
              {captain ? <td>{(player.predicted_points * 2).toFixed(2)}</td> : null}
              <td>{player.valueScore.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </DataTable>
  );
}

function Progress({ value, max }: { value: number; max: number }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="progress-cell">
      <span>{value.toFixed(2)}</span>
      <div className="progress"><div style={{ width: `${pct}%` }} /></div>
    </div>
  );
}

function Dashboard({ players }: { players: PlayerView[] }) {
  const currentGw = players[0]?.gameweek ?? "-";
  const top = players.slice(0, 5);
  const captain = players[0];

  return (
    <>
      <PageHeader
        kicker={`Gameweek ${currentGw} · Active`}
        title="FPL Intelligence Dashboard"
        subtitle="Machine-learning powered predictions, refreshed from Supabase after the Python pipeline runs."
        badges={["XGBoost", "Supabase", "FPL API"]}
      />

      <section className="stat-grid">
        <StatCard label="Current Gameweek" value={`GW ${currentGw}`} sub="Live model board" />
        <StatCard label="Players Tracked" value={players.length.toLocaleString()} sub="Premier League pool" />
        <StatCard label="Captain Pick" value={captain?.player_name ?? "-"} sub={captain ? `${captain.predicted_points.toFixed(2)} xPts` : undefined} />
      </section>

      <h2>Top predicted by position</h2>
      <section className="card-grid four">
        {positions.map((position) => {
          const player = topByPosition(players, position);
          return player ? <PlayerCard key={position} player={player} rank={position} /> : null;
        })}
      </section>

      <h2>Top 5 overall</h2>
      <PlayerTable players={top} />
    </>
  );
}

function Picks({ players }: { players: PlayerView[] }) {
  const [position, setPosition] = useState("ALL");
  const [maxPrice, setMaxPrice] = useState(15);
  const filtered = players
    .filter((player) => position === "ALL" || player.position === position)
    .filter((player) => player.price <= maxPrice)
    .slice(0, 50);

  return (
    <>
      <PageHeader kicker="GW Picks" title="Predicted points board" subtitle="Filter the live model rankings by position and budget." badges={["xPts", "Value", "Price"]} />
      <div className="toolbar">
        <select value={position} onChange={(event) => setPosition(event.target.value)}>
          <option value="ALL">All positions</option>
          {positions.map((pos) => <option key={pos}>{pos}</option>)}
        </select>
        <label>
          Max price {formatPrice(maxPrice)}
          <input type="range" min="4" max="15" step="0.5" value={maxPrice} onChange={(event) => setMaxPrice(Number(event.target.value))} />
        </label>
      </div>
      <section className="card-grid three">
        {filtered.slice(0, 6).map((player, index) => <PlayerCard key={player.player_id} player={player} rank={index + 1} />)}
      </section>
      <PlayerTable players={filtered} />
    </>
  );
}

function Captain({ players }: { players: PlayerView[] }) {
  const top = players.slice(0, 10);
  return (
    <>
      <PageHeader kicker="Captain" title="Captain picks" subtitle="Doubled predicted points ranking." badges={["2x scoring", "Top 10"]} />
      <section className="card-grid two">
        {top.slice(0, 4).map((player, index) => <PlayerCard key={player.player_id} player={player} rank={index + 1} captain />)}
      </section>
      <PlayerTable players={top} captain />
    </>
  );
}

function Prices({ players }: { players: PlayerView[] }) {
  const value = [...players].sort((a, b) => b.valueScore - a.valueScore).slice(0, 30);
  return (
    <>
      <PageHeader kicker="Value" title="Predicted points per pound" subtitle="Find budget-efficient picks from the live prediction table." badges={["Value", "Budget"]} />
      <PlayerTable players={value} />
    </>
  );
}

function Optimizer({ players }: { players: PlayerView[] }) {
  const [budget, setBudget] = useState(100);
  const [maxPerTeam, setMaxPerTeam] = useState(3);
  const squad = useMemo(() => pickSquad(players, budget, maxPerTeam), [players, budget, maxPerTeam]);
  const totalCost = squad.reduce((sum, player) => sum + player.price, 0);
  const totalPoints = squad.reduce((sum, player) => sum + player.predicted_points, 0);

  return (
    <>
      <PageHeader kicker="Optimizer" title="Squad optimizer" subtitle="A fast browser-side squad builder using the prediction board." badges={["15 players", "Budget", "Team cap"]} />
      <div className="toolbar">
        <label>Budget {formatPrice(budget)}<input type="range" min="75" max="110" step="0.5" value={budget} onChange={(event) => setBudget(Number(event.target.value))} /></label>
        <label>Max per team {maxPerTeam}<input type="range" min="1" max="3" step="1" value={maxPerTeam} onChange={(event) => setMaxPerTeam(Number(event.target.value))} /></label>
      </div>
      <section className="stat-grid">
        <StatCard label="Squad Cost" value={formatPrice(totalCost)} sub={`Budget ${formatPrice(budget)}`} />
        <StatCard label="Total xPts" value={totalPoints.toFixed(1)} sub={`${squad.length}/15 selected`} />
      </section>
      <PlayerTable players={squad} />
    </>
  );
}

type FixtureRow = {
  team: string;
  opponent: string;
  gameweek: number;
  fdr: number;
  venue: "H" | "A";
};

function Fixtures() {
  const [rows, setRows] = useState<FixtureRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [bootstrapRes, fixturesRes] = await Promise.all([
          fetch("https://fantasy.premierleague.com/api/bootstrap-static/"),
          fetch("https://fantasy.premierleague.com/api/fixtures/"),
        ]);
        const bootstrap = await bootstrapRes.json();
        const fixtures = await fixturesRes.json();
        const teams = new Map<number, string>(bootstrap.teams.map((team: { id: number; short_name: string }) => [team.id, team.short_name]));
        const current = bootstrap.events.find((event: { is_current: boolean }) => event.is_current)?.id ?? 1;
        const nextRows: FixtureRow[] = [];

        fixtures
          .filter((fixture: { event: number | null }) => fixture.event && fixture.event >= current && fixture.event < current + 6)
          .forEach((fixture: { event: number; team_h: number; team_a: number; team_h_difficulty: number; team_a_difficulty: number }) => {
            nextRows.push({
              team: teams.get(fixture.team_h) ?? "UNK",
              opponent: teams.get(fixture.team_a) ?? "UNK",
              gameweek: fixture.event,
              fdr: fixture.team_h_difficulty,
              venue: "H",
            });
            nextRows.push({
              team: teams.get(fixture.team_a) ?? "UNK",
              opponent: teams.get(fixture.team_h) ?? "UNK",
              gameweek: fixture.event,
              fdr: fixture.team_a_difficulty,
              venue: "A",
            });
          });

        setRows(nextRows);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not load fixture data");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const grouped = useMemo(() => {
    const map = new Map<string, FixtureRow[]>();
    rows.forEach((row) => {
      map.set(row.team, [...(map.get(row.team) ?? []), row]);
    });
    return [...map.entries()]
      .map(([team, fixtures]) => ({ team, fixtures, avg: fixtures.reduce((sum, row) => sum + row.fdr, 0) / fixtures.length }))
      .sort((a, b) => a.avg - b.avg);
  }, [rows]);

  return (
    <>
      <PageHeader kicker="Fixtures" title="Fixture difficulty" subtitle="Teams sorted by easiest upcoming fixture run." badges={["Next 6 GWs", "FDR"]} />
      {loading ? <div className="stat-card">Loading fixtures...</div> : null}
      {error ? <div className="stat-card">{error}</div> : null}
      {!loading && !error ? (
        <DataTable>
          <table>
            <thead>
              <tr>
                <th>Team</th>
                <th>Upcoming fixtures</th>
                <th>Avg FDR</th>
              </tr>
            </thead>
            <tbody>
              {grouped.map((row) => (
                <tr key={row.team}>
                  <td>{row.team}</td>
                  <td>
                    <div className="fixture-strip">
                      {row.fixtures.map((fixture) => (
                        <span className={`fdr fdr-${Math.round(fixture.fdr)}`} key={`${row.team}-${fixture.gameweek}-${fixture.opponent}-${fixture.venue}`}>
                          GW{fixture.gameweek} {fixture.opponent} ({fixture.venue})
                        </span>
                      ))}
                    </div>
                  </td>
                  <td>{row.avg.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </DataTable>
      ) : null}
    </>
  );
}

function MyTeam({ players }: { players: PlayerView[] }) {
  const [teamId, setTeamId] = useState("");
  const [squad, setSquad] = useState<PlayerView[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function loadTeam() {
    if (!teamId.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const bootstrapRes = await fetch("https://fantasy.premierleague.com/api/bootstrap-static/");
      const bootstrap = await bootstrapRes.json();
      const current = bootstrap.events.find((event: { is_current: boolean }) => event.is_current)?.id ?? 1;
      const picksRes = await fetch(`https://fantasy.premierleague.com/api/entry/${teamId}/event/${current}/picks/`);
      if (!picksRes.ok) throw new Error("Could not load this FPL team. Check the team ID.");
      const picks = await picksRes.json();
      const ids = new Set<number>(picks.picks.map((pick: { element: number }) => pick.element));
      setSquad(players.filter((player) => ids.has(player.player_id)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load team");
    } finally {
      setLoading(false);
    }
  }

  const weakLinks = [...squad].sort((a, b) => a.predicted_points - b.predicted_points).slice(0, 3);

  return (
    <>
      <PageHeader kicker="My Team" title="Your squad analysis" subtitle="Load your FPL team and compare your players against the prediction board." badges={["Team ID", "Transfers"]} />
      <div className="toolbar">
        <input value={teamId} onChange={(event) => setTeamId(event.target.value)} placeholder="FPL Team ID" />
        <button className="primary-button" onClick={loadTeam}>{loading ? "Loading..." : "Load team"}</button>
      </div>
      {error ? <div className="stat-card">{error}</div> : null}
      {squad.length ? (
        <>
          <PlayerTable players={squad.sort((a, b) => b.predicted_points - a.predicted_points)} />
          <h2>Suggested upgrade search</h2>
          <section className="card-grid three">
            {weakLinks.map((player) => {
              const replacement = players.find((candidate) =>
                candidate.position === player.position &&
                candidate.player_id !== player.player_id &&
                candidate.price <= player.price + 1.5 &&
                candidate.predicted_points > player.predicted_points
              );
              return (
                <div className="stat-card" key={player.player_id}>
                  <div className="stat-label">Consider replacing</div>
                  <div className="stat-value">{player.player_name}</div>
                  <div className="stat-sub">
                    {replacement ? `Target ${replacement.player_name} (+${(replacement.predicted_points - player.predicted_points).toFixed(2)} xPts)` : "No clear upgrade in range"}
                  </div>
                </div>
              );
            })}
          </section>
        </>
      ) : null}
    </>
  );
}

function Placeholder({ title, kicker }: { title: string; kicker: string }) {
  return (
    <PageHeader
      kicker={kicker}
      title={title}
      subtitle="This React view is ready for the next API/data integration step."
      badges={["React", "Supabase"]}
    />
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const { predictions, stats, loading, error } = useFplData();
  const players = useMemo(() => buildPlayerViews(predictions, stats), [predictions, stats]);

  let content;
  if (!hasSupabaseConfig) {
    content = <PageHeader kicker="Configuration" title="Supabase env missing" subtitle="Add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to web/.env." />;
  } else if (loading) {
    content = <PageHeader kicker="Loading" title="Fetching FPL intelligence" subtitle="Reading predictions and player stats from Supabase." />;
  } else if (error) {
    content = <PageHeader kicker="Error" title="Could not load data" subtitle={error} />;
  } else if (!players.length) {
    content = <PageHeader kicker="No data" title="No predictions found" subtitle="Run the Python pipeline so Supabase has prediction rows." />;
  } else {
    content = {
      dashboard: <Dashboard players={players} />,
      picks: <Picks players={players} />,
      captain: <Captain players={players} />,
      fixtures: <Fixtures />,
      prices: <Prices players={players} />,
      optimizer: <Optimizer players={players} />,
      team: <MyTeam players={players} />,
    }[page];
  }

  return <Shell page={page} setPage={setPage}>{content}</Shell>;
}
