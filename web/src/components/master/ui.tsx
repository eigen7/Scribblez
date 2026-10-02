import { ReactNode } from 'react';
import { relTime } from './MasterApp';

// Shared master-dashboard primitives: KPI stat tiles, worker-health badges,
// tag links, state colors, and compact-count and byte formatting. Styled by the .tile / .pill-stale / .dot-ok
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

// Text colors for tag, slot and machine states; a state not listed is shown
// in the body color.
export const stateColors: Record<string, string> = {
  running: '#2a7a2a', paused: '#8494a5', exited: '#b23b3b', failed: '#b23b3b', finished: '#446e9b',
  complete: '#446e9b', idle: '#8494a5',
  up: '#2a7a2a', 'no docker': '#b23b3b', launching: '#1f77b4', preparing: '#1f77b4',
  stopped: '#8494a5', gone: '#b23b3b',
  waiting: '#a05a00',
  unreachable: '#a05a00',
  starting: '#1f77b4', stopping: '#1f77b4',
  // No reconcile pass has observed this slot yet (only that pass talks to the
  // machines; a status request reads what it left behind).
  checking: '#8494a5',
};

// A state in its color, bold: a tag's state, a slot's, a machine's.
export function StateText({ state }: { state: string }) {
  return <span style={{ color: stateColors[state] ?? '#1a1f28', fontWeight: 600 }}>{state}</span>;
}

// Disk sizes as du -h prints them, in powers of 1024: 29097127936 -> 27.1G.
export function fmtBytes(v: number): string {
  const units = ['B', 'K', 'M', 'G', 'T'];
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return i === 0 ? `${v}B` : `${v.toFixed(1)}${units[i]}`;
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

// A workload/tag name that opens the tag's task view.
export function TagLink({ workload, tag, onOpen, title }: {
  workload: string; tag: string; onOpen: (workload: string, tag: string) => void; title?: string;
}) {
  return (
    <span
      onClick={() => onOpen(workload, tag)}
      title={title ?? `open ${workload}/${tag}`}
      style={{ color: '#1f77b4', cursor: 'pointer' }}
    >
      {workload}/{tag}
    </span>
  );
}
