import { ReactNode, useEffect, useRef, useState } from 'react';
import { getJSON } from '../../lib/api';
import BokehFigure from '../BokehFigure';
import { HealthBadge, Tile, fmtCompact, isStale } from './ui';

// The generic worker Stats tab, driven entirely by the stats schema the API
// reports (each role's unit noun and timing phases), so any workload role
// that publishes stats renders here without workload-specific code. Per
// role: a row of fleet-aggregate tiles, the rate and cycle-breakdown
// figures side by side, and a compact per-worker detail table.

type RoleStats = { title: string; unit: string; phases: Record<string, string> };
type WorkerRow = {
  worker_id: string; role: string | null; kind: string; threads: number | null;
  bundle_id: string | null; host_arch: string | null; bundle_arch: string | null;
  units_total: number; cycles_total: number; updated_at: number;
  units_per_hour: number | null; phases: Record<string, number>; upload_mbps: number | null;
};
type StatsPayload = { roles: Record<string, RoleStats>; workers: WorkerRow[]; updated_at: number };

const fmt = (v: number | null | undefined, digits = 1) => (v == null ? '—' : v.toFixed(digits));

const cycleSeconds = (w: WorkerRow) => Object.values(w.phases).reduce((a, b) => a + b, 0);
const workerStale = (w: WorkerRow) => isStale(w.updated_at, cycleSeconds(w));

function StatsFigure({ workload, tag, role, name, version }: {
  workload: string; tag: string; role: string; name: string; version: number;
}) {
  const [item, setItem] = useState<unknown | null>(null);
  useEffect(() => {
    getJSON(
      `/api/task/figure/${name}?workload=${workload}&tag=${encodeURIComponent(tag)}&role=${role}`,
    )
      .then((d) => setItem(d.item ?? null))
      .catch(() => setItem(null));
  }, [workload, tag, role, name, version]);
  if (!item) return null;
  return <div className="card"><BokehFigure item={item} /></div>;
}

// Fleet aggregates for one role's tile row. Stale workers are excluded from
// the rates and means so a dead worker's last recorded window doesn't keep
// inflating the fleet numbers.
function aggregates(workers: WorkerRow[]) {
  const fresh = workers.filter((w) => !workerStale(w));
  const rates = fresh.map((w) => w.units_per_hour).filter((v): v is number => v != null);
  const uploads = fresh.map((w) => w.upload_mbps).filter((v): v is number => v != null);
  return {
    fresh,
    staleCount: workers.length - fresh.length,
    rate: rates.length ? rates.reduce((a, b) => a + b, 0) : null,
    total: workers.reduce((a, w) => a + w.units_total, 0),
    cycle: fresh.length
      ? fresh.reduce((a, w) => a + cycleSeconds(w), 0) / fresh.length
      : null,
    upload: uploads.length ? uploads.reduce((a, b) => a + b, 0) : null,
  };
}

// "self-play 1.0 · deliver 0.1": the cycle tile's per-phase mean split.
function phaseSplit(stats: RoleStats, fresh: WorkerRow[]): string {
  if (fresh.length === 0) return '';
  return Object.entries(stats.phases)
    .map(([key, label]) => {
      const mean = fresh.reduce((a, w) => a + (w.phases[key] ?? 0), 0) / fresh.length;
      return `${label} ${fmt(mean)}`;
    })
    .join(' · ');
}

// One column of the per-worker table: header, sort key, numeric alignment,
// and an optional display renderer (the sort key formatted, by default).
type Col = {
  header: string;
  key: (w: WorkerRow) => string | number | null;
  numeric?: boolean;
  render?: (w: WorkerRow) => ReactNode;
};

const dim = (s: string) => <span className="dim">{s}</span>;

function defaultRender(col: Col, w: WorkerRow): ReactNode {
  const v = col.key(w);
  if (v == null) return dim('—');
  return typeof v === 'number' && !Number.isInteger(v) ? fmt(v) : String(v);
}

// A bundle id is "<git-sha-12>[-dirty]-<content-hash-8>"; the table shows the
// git half (what identifies the code) and carries the whole id as a tooltip.
// A local worker runs the controller's own tree and has no bundle.
function bundleLabel(w: WorkerRow): ReactNode {
  if (w.bundle_id == null) return dim('—');
  const [sha, dirty] = w.bundle_id.split('-');
  return <span title={w.bundle_id}>{sha}{dirty === 'dirty' ? '-dirty' : ''}</span>;
}

function archLabel(w: WorkerRow): ReactNode {
  if (w.host_arch == null) return dim('—');
  const runs = w.bundle_arch && w.bundle_arch !== w.host_arch ? ` (runs ${w.bundle_arch})` : '';
  return `${w.host_arch}${runs}`;
}

function columns(stats: RoleStats): Col[] {
  return [
    { header: 'worker', key: (w) => w.worker_id },
    { header: 'kind', key: (w) => w.kind },
    { header: 'threads', key: (w) => w.threads, numeric: true },
    { header: 'bundle', key: (w) => w.bundle_id, render: bundleLabel },
    { header: 'arch', key: (w) => w.host_arch, render: archLabel },
    {
      header: stats.unit, key: (w) => w.units_total, numeric: true,
      render: (w) => fmtCompact(w.units_total),
    },
    {
      // A stale worker's rate is a reading of a window that ended long ago,
      // so it shows as absent rather than as a misleading number.
      header: `${stats.unit}/hr`, key: (w) => w.units_per_hour, numeric: true,
      render: (w) => (workerStale(w) ? dim('—') : fmtCompact(w.units_per_hour)),
    },
    ...Object.entries(stats.phases).map(([key, label]): Col => ({
      header: `${label} s`, key: (w) => w.phases[key] ?? null, numeric: true,
    })),
    { header: 'upload MB/s', key: (w) => w.upload_mbps, numeric: true },
    {
      header: 'updated', key: (w) => w.updated_at,
      render: (w) => <HealthBadge updatedAt={w.updated_at} stale={workerStale(w)} />,
    },
  ];
}

function WorkerTable({ stats, workers }: { stats: RoleStats; workers: WorkerRow[] }) {
  const [sortCol, setSortCol] = useState(0);
  const [sortAsc, setSortAsc] = useState(true);
  const cols = columns(stats);
  const key = cols[sortCol]?.key ?? cols[0].key;
  const sorted = [...workers].sort((a, b) => {
    const [ka, kb] = [key(a), key(b)];
    if (ka == null || kb == null) return (ka == null ? 1 : 0) - (kb == null ? 1 : 0); // nulls last
    const cmp = typeof ka === 'string' ? ka.localeCompare(String(kb)) : Number(ka) - Number(kb);
    return sortAsc ? cmp : -cmp;
  });
  const clickHeader = (i: number) => {
    if (i === sortCol) setSortAsc((v) => !v);
    else { setSortCol(i); setSortAsc(true); }
  };

  return (
    <div className="card" style={{ overflowX: 'auto' }}>
      <table className="data-table">
        <thead>
          <tr>
            {cols.map((c, i) => (
              <th
                key={c.header}
                className={c.numeric ? 'num' : undefined}
                onClick={() => clickHeader(i)}
                style={{ cursor: 'pointer', userSelect: 'none' }}
              >
                {c.header}{i === sortCol ? (sortAsc ? ' ▲' : ' ▼') : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((w) => (
            <tr key={w.worker_id}>
              {cols.map((c) => (
                <td key={c.header} className={c.numeric ? 'num' : undefined}>
                  {c.render ? c.render(w) : defaultRender(c, w)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RoleSection({ workload, tag, role, stats, workers, version }: {
  workload: string; tag: string; role: string; stats: RoleStats;
  workers: WorkerRow[]; version: number;
}) {
  const agg = aggregates(workers);
  return (
    <div className="role-section">
      <div className="role-head">
        <b>{stats.title}s</b>
        <span>{workers.length} worker{workers.length === 1 ? '' : 's'}</span>
      </div>
      <div className="tiles">
        <Tile
          label={`${stats.unit} / hr`}
          value={fmtCompact(agg.rate)}
          sub="recent window"
        />
        <Tile label={`${stats.unit} total`} value={fmtCompact(agg.total)} />
        <Tile
          label="cycle time"
          value={fmt(agg.cycle)}
          unit={agg.cycle == null ? undefined : 's'}
          sub={phaseSplit(stats, agg.fresh)}
        />
        {agg.upload != null && <Tile label="upload" value={fmt(agg.upload)} unit="MB/s" />}
        <Tile
          label="workers"
          value={String(workers.length)}
          sub={agg.staleCount > 0 ? (
            <span className="pill-stale">{agg.staleCount} stale</span>
          ) : (
            <span className="health-ok"><span className="dot-ok" /> all fresh</span>
          )}
        />
      </div>
      <div className="charts-grid">
        <StatsFigure workload={workload} tag={tag} role={role} name="rate" version={version} />
        <StatsFigure
          workload={workload} tag={tag} role={role} name="cycle_breakdown" version={version}
        />
      </div>
      <WorkerTable stats={stats} workers={workers} />
    </div>
  );
}

export default function StatsTab({ workload, tag }: { workload: string; tag: string }) {
  const [data, setData] = useState<StatsPayload | null>(null);
  const [version, setVersion] = useState(0);
  const lastUpdate = useRef(0);

  useEffect(() => {
    let live = true;
    const refresh = async () => {
      try {
        const d: StatsPayload = await getJSON(
          `/api/task/stats?workload=${workload}&tag=${encodeURIComponent(tag)}`,
        );
        if (!live) return;
        setData(d);
        if (d.updated_at !== lastUpdate.current) {
          lastUpdate.current = d.updated_at;
          setVersion((v) => v + 1); // stats advanced -> re-fetch the figures
        }
      } catch { /* API may be restarting; the poll retries */ }
    };
    refresh();
    const id = setInterval(refresh, 5000);
    return () => { live = false; clearInterval(id); };
  }, [workload, tag]);

  if (!data || data.workers.length === 0) {
    return (
      <div className="card" style={{ color: '#556070', fontStyle: 'italic' }}>
        No worker stats yet — they appear after each worker's first completed cycle.
      </div>
    );
  }

  return (
    <>
      {Object.entries(data.roles).map(([role, stats]) => {
        const workers = data.workers.filter((w) => w.role === role);
        if (workers.length === 0) return null;
        return (
          <RoleSection
            key={role} workload={workload} tag={tag} role={role} stats={stats}
            workers={workers} version={version}
          />
        );
      })}
    </>
  );
}
