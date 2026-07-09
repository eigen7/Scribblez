import { useEffect, useRef, useState } from 'react';
import { getJSON } from '../../lib/api';
import BokehFigure from '../BokehFigure';
import { relTime } from './MasterApp';

// The generic worker Stats tab: per-role summary tables and figures, driven
// entirely by the stats schema the API reports (each role's unit noun and
// timing phases), so any workload role that publishes stats renders here
// without workload-specific code.

type RoleStats = { title: string; unit: string; phases: Record<string, string> };
type WorkerRow = {
  worker_id: string; role: string | null; kind: string; threads: number | null;
  host_arch: string | null; bundle_arch: string | null;
  units_total: number; cycles_total: number; updated_at: number;
  units_per_hour: number | null; phases: Record<string, number>; upload_mbps: number | null;
};
type StatsPayload = { roles: Record<string, RoleStats>; workers: WorkerRow[]; updated_at: number };

const fmt = (v: number | null | undefined, digits = 1, suffix = '') =>
  v == null ? '—' : `${v.toFixed(digits)}${suffix}`;

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
  return <div style={{ marginTop: 12 }}><BokehFigure item={item} /></div>;
}

// Column headers + sort keys for one role's table, derived from its phase schema.
function columns(stats: RoleStats): [string, (w: WorkerRow) => string | number | null][] {
  return [
    ['worker', (w) => w.worker_id],
    ['kind', (w) => w.kind],
    ['threads', (w) => w.threads],
    ['arch', (w) => w.host_arch],
    [stats.unit, (w) => w.units_total],
    [`${stats.unit}/hr`, (w) => w.units_per_hour],
    ...Object.entries(stats.phases).map(
      ([key, label]): [string, (w: WorkerRow) => number | null] => [
        `${label} s`, (w) => w.phases[key] ?? null,
      ],
    ),
    ['upload MB/s', (w) => w.upload_mbps],
    ['updated', (w) => w.updated_at],
  ];
}

function RoleSection({ workload, tag, role, stats, workers, version }: {
  workload: string; tag: string; role: string; stats: RoleStats;
  workers: WorkerRow[]; version: number;
}) {
  const [sortCol, setSortCol] = useState(0);
  const [sortAsc, setSortAsc] = useState(true);
  const cols = columns(stats);
  const key = cols[sortCol]?.[1] ?? cols[0][1];
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

  const cell = (w: WorkerRow, i: number): string => {
    const v = cols[i][1](w);
    if (v == null) return '—';
    if (cols[i][0] === 'updated') return relTime(Number(v));
    if (cols[i][0] === 'arch') {
      return `${w.host_arch}${w.bundle_arch && w.bundle_arch !== w.host_arch ? ` (runs ${w.bundle_arch})` : ''}`;
    }
    return typeof v === 'number' && !Number.isInteger(v) ? fmt(v) : String(v);
  };

  return (
    <div className="card" style={{ marginBottom: 14 }}>
      <b style={{ fontSize: 15 }}>{stats.title}s</b>
      <table style={{ borderCollapse: 'collapse', fontSize: 13, width: '100%', marginTop: 8 }}>
        <thead>
          <tr style={{ textAlign: 'left', color: '#445063' }}>
            {cols.map(([h], i) => (
              <th
                key={h}
                onClick={() => clickHeader(i)}
                style={{ padding: '4px 12px 4px 0', cursor: 'pointer', userSelect: 'none' }}
              >
                {h}{i === sortCol ? (sortAsc ? ' ▲' : ' ▼') : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((w) => (
            <tr key={w.worker_id} style={{ borderTop: '1px solid #e2e8ee' }}>
              {cols.map(([h], i) => (
                <td key={h} style={{ padding: '5px 12px 5px 0', fontWeight: i === 0 ? 600 : 400 }}>
                  {cell(w, i)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <StatsFigure workload={workload} tag={tag} role={role} name="timeline" version={version} />
      <StatsFigure workload={workload} tag={tag} role={role} name="throughput" version={version} />
      <StatsFigure workload={workload} tag={tag} role={role} name="cycle_breakdown" version={version} />
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
