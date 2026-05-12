import { BarChart3, CalendarDays, Crown, Flame, Gauge, LayoutDashboard, PiggyBank, UserRound, Users } from "lucide-react";
import type { ReactNode } from "react";

type Page = "dashboard" | "picks" | "captain" | "fixtures" | "prices" | "optimizer" | "team";

type ShellProps = {
  page: Page;
  setPage: (page: Page) => void;
  children: ReactNode;
};

const nav = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "picks", label: "GW Picks", icon: BarChart3 },
  { id: "captain", label: "Captain", icon: Crown },
  { id: "fixtures", label: "Fixtures", icon: CalendarDays },
  { id: "prices", label: "Prices", icon: PiggyBank },
  { id: "optimizer", label: "Optimizer", icon: Users },
  { id: "team", label: "My Team", icon: UserRound },
] as const;

export function Shell({ page, setPage, children }: ShellProps) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Gauge size={20} /></div>
          <div>
            <div className="brand-title">FPL Intelligence</div>
            <div className="brand-subtitle">ML predictions</div>
          </div>
        </div>

        <nav className="nav-list">
          {nav.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={`nav-item ${page === item.id ? "active" : ""}`}
                onClick={() => setPage(item.id)}
              >
                <Icon size={16} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-card">
          <Flame size={16} />
          <span>Powered by Supabase + XGBoost</span>
        </div>
      </aside>
      <main className="main-panel">{children}</main>
    </div>
  );
}
