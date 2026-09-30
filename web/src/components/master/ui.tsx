import { ReactNode } from 'react';
import { relTime } from './MasterApp';

// Shared master-dashboard primitives: KPI stat tiles, worker-health badges,
// and compact-count formatting. Styled by the .tile / .pill-stale / .dot-ok
// classes in index.css.

export function Tile({ label, value, unit, sub }: {
  label: string; value: string; unit?: string; sub?: ReactNode;
}) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}{unit ? <small> {unit}</small> : null}</div>
      {sub != null && <div className="tile-sub">{sub}</div>}
    </div>
  );
}

// Compact display for large counts and rates: 63000 -> 63.0K, 3542356 -> 3.54M.
export function fmtCompact(v: number | null | undefined): string {
  if (v == null) return '—';
  if (Math.abs(v) >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (Math.abs(v) >= 1e4) return `${(v / 1e3).toFixed(1)}K`;
  return String(Math.round(v));
}

export function HealthBadge({ updatedAt, stale }: { updatedAt: number; stale: boolean }) {
  if (stale) return <span className="pill-stale">stale · {relTime(updatedAt)}</span>;
  return (
    <span className="health-ok">
      <span className="dot-ok" /> {relTime(updatedAt)}
    </span>
  );
}
