import { useCallback, useEffect, useRef, useState } from 'react';
import { getJSON, postJSON } from '../../lib/api';
import BokehFigure from '../BokehFigure';
import { Button, relTime, Workload } from './MasterApp';

// One task's view in the master dashboard: an Overview tab (frozen params,
// general task info, the worker slots with add/pause/start/remove controls)
// plus workload-specific tabs (kill_test: a Stats tab of per-worker
// throughput/bottleneck figures built from the workers' stats records).

type WorkerInfo = {
  worker_id: string; kind: 'local' | 'cloud'; desired_state: string; state: string;
  threads: number | null; vcpus: number | null; flavor: string | null;
  pod_id: string | null; cost_per_hr?: number; public_ip?: string; ssh?: string;
};
type TaskInfo = {
  workload: string; tag: string; has_task: boolean; params: Record<string, number | boolean> | null;
  created_at: number | null; pairs: number; data_dir: string; workers: WorkerInfo[];
  spend: number;
};
type WorkerStats = {
  worker_id: string; kind: string; threads: number | null; host_arch: string | null;
  bundle_arch: string | null; pairs_total: number; cycles_total: number; updated_at: number;
  pairs_per_hour: number | null; gen_s: number; sim_s: number; upload_s: number;
  upload_mbps: number | null;
};

const stateColors: Record<string, string> = {
  running: '#2a7a2a', paused: '#8494a5', exited: '#b23b3b',
  interrupted: '#a05a00', terminated: '#b23b3b',
};

// The Runpod CPU flavor ids accepted by pod creation (the REST API's fixed
// enum -- there is no listing endpoint): generation 3/5, compute- / general- /
// memory-optimized. Live availability and pricing are only visible in the
// Runpod console.
const CPU_FLAVORS = ['cpu3c', 'cpu3g', 'cpu3m', 'cpu5c', 'cpu5g', 'cpu5m'];
const RUNPOD_CONSOLE = 'https://console.runpod.io/deploy';

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card" style={{ marginBottom: 14 }}>
      <b style={{ fontSize: 15 }}>{title}</b>
      <div style={{ marginTop: 8 }}>{children}</div>
    </div>
  );
}

function KV({ items }: { items: [string, React.ReactNode][] }) {
  return (
    <table style={{ fontSize: 14, borderCollapse: 'collapse' }}>
      <tbody>
        {items.map(([k, v]) => (
          <tr key={k}>
            <td style={{ color: '#445063', padding: '2px 18px 2px 0' }}>{k}</td>
            <td style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const numInput = { fontSize: 14, padding: '3px 6px', border: '1px solid #b8c4d0', borderRadius: 4, width: 70 };

function AddWorkerForms({ workload, tag, onError, onChanged }: {
  workload: Workload; tag: string; onError: (e: string) => void; onChanged: () => void;
}) {
  const [threads, setThreads] = useState('');
  const [count, setCount] = useState('1');
  const [vcpus, setVcpus] = useState('16');
  const [flavor, setFlavor] = useState('cpu3c');
  // Which form is mid-request ('local' | 'cloud' | null): its button shows a
  // progress label -- pod creation takes a few seconds.
  const [busy, setBusy] = useState<'local' | 'cloud' | null>(null);

  const add = async (kind: 'local' | 'cloud', body: Record<string, unknown>) => {
    setBusy(kind);
    onError('');
    try {
      await postJSON('/api/task/workers', { workload: workload.name, tag, kind, ...body });
      onChanged();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div style={{ display: 'flex', gap: 40, flexWrap: 'wrap', marginTop: 12 }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
        <label style={{ fontSize: 13 }}>
          Local worker — threads<br />
          <input
            style={numInput} value={threads} placeholder="all"
            onChange={(e) => setThreads(e.target.value)}
          />
        </label>
        <Button
          label={busy === 'local' ? 'Adding…' : 'Add local'}
          disabled={busy !== null}
          onClick={() => add('local', { threads: threads ? parseInt(threads, 10) : null })}
        />
      </div>
      <div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
          <label style={{ fontSize: 13 }}>
            Cloud workers — count<br />
            <input style={numInput} value={count} onChange={(e) => setCount(e.target.value)} />
          </label>
          <label style={{ fontSize: 13 }}>
            vCPUs<br />
            <input style={numInput} value={vcpus} onChange={(e) => setVcpus(e.target.value)} />
          </label>
          <label style={{ fontSize: 13 }}>
            flavor<br />
            <select value={flavor} onChange={(e) => setFlavor(e.target.value)} style={{ fontSize: 14 }}>
              {CPU_FLAVORS.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
          </label>
          <Button
            label={busy === 'cloud' ? 'Creating pods…' : 'Add cloud'}
            disabled={busy !== null}
            onClick={() => add('cloud', {
              count: parseInt(count, 10) || 1, vcpus: parseInt(vcpus, 10) || 16, flavor,
            })}
          />
        </div>
        <div style={{ fontSize: 12, color: '#556070', marginTop: 6 }}>
          flavors: cpu3/cpu5 = hardware generation; c/g/m = compute/general/memory-optimized.{' '}
          <a href={RUNPOD_CONSOLE} target="_blank" rel="noreferrer">availability &amp; pricing ↗</a>
          {workload.interruptible &&
            ' — rented interruptible (discounted; auto-restarted if reclaimed)'}
        </div>
      </div>
    </div>
  );
}

function WorkersTable({ workers, onAction }: {
  workers: WorkerInfo[]; onAction: (workerId: string, action: string) => void;
}) {
  if (workers.length === 0) {
    return <div style={{ color: '#556070', fontStyle: 'italic' }}>No workers yet.</div>;
  }
  return (
    <table style={{ borderCollapse: 'collapse', fontSize: 14, width: '100%' }}>
      <thead>
        <tr style={{ textAlign: 'left', color: '#445063' }}>
          {['worker', 'kind', 'resources', 'state', '$/hr', 'connect', ''].map((h) => (
            <th key={h} style={{ padding: '4px 14px 4px 0' }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {workers.map((w) => {
          const running = w.state === 'running';
          return (
            <tr key={w.worker_id} style={{ borderTop: '1px solid #e2e8ee' }}>
              <td style={{ padding: '6px 14px 6px 0', fontWeight: 600 }}>{w.worker_id}</td>
              <td style={{ padding: '6px 14px 6px 0' }}>{w.kind}</td>
              <td style={{ padding: '6px 14px 6px 0' }}>
                {w.kind === 'local' ? `${w.threads} threads` : `${w.vcpus} vcpu ${w.flavor}`}
              </td>
              <td style={{ padding: '6px 14px 6px 0', color: stateColors[w.state] ?? '#1a1f28', fontWeight: 600 }}>
                {w.state}
              </td>
              <td style={{ padding: '6px 14px 6px 0' }}>
                {w.cost_per_hr != null ? `$${w.cost_per_hr}` : '—'}
              </td>
              <td style={{ padding: '6px 14px 6px 0', fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                {w.ssh ?? '—'}
              </td>
              <td style={{ padding: '6px 0', whiteSpace: 'nowrap', display: 'flex', gap: 6 }}>
                {running
                  ? <Button label="Pause" onClick={() => onAction(w.worker_id, 'pause')} />
                  : <Button label="Start" onClick={() => onAction(w.worker_id, 'start')} />}
                <span title={running ? 'pause the worker before removing it' : undefined}>
                  <Button
                    label="Remove" tone="danger" disabled={running}
                    onClick={() => onAction(w.worker_id, 'remove')}
                  />
                </span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function OverviewTab({ workload, tag }: { workload: Workload; tag: string }) {
  const [info, setInfo] = useState<TaskInfo | null>(null);
  const [error, setError] = useState('');

  const refresh = useCallback(() => {
    getJSON(`/api/task?workload=${workload.name}&tag=${encodeURIComponent(tag)}`)
      .then(setInfo)
      .catch((e) => setError(String(e)));
  }, [workload, tag]);
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [refresh]);

  if (!info) return <div className="card">Loading…</div>;

  const act = async (body: Record<string, unknown>) => {
    setError('');
    try {
      await postJSON('/api/task/worker_action', { workload: workload.name, tag, ...body });
      refresh();
    } catch (e) {
      setError(String(e));
    }
  };
  const cloudCost = info.workers.reduce((s, w) => s + (w.state === 'running' ? w.cost_per_hr ?? 0 : 0), 0);
  const anyRunning = info.workers.some((w) => w.state === 'running');
  const anyStartable = info.workers.some((w) => w.state !== 'running');

  return (
    <>
      {!info.has_task && (
        <div className="card" style={{ color: '#a05a00', marginBottom: 14 }}>
          This tag predates the dashboard (no task record): data is viewable but workers can't
          be attached. Generate into it with the CLI, or create a new tag.
        </div>
      )}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
        <Card title="Task">
          <KV items={[
            ['workload', workload.title],
            ['created', info.created_at ? new Date(info.created_at * 1000).toLocaleString() : '—'],
            ['complete pairs', info.pairs],
            ['data dir', info.data_dir],
            ['cloud burn rate', `$${cloudCost.toFixed(3)}/hr`],
            ['cloud spend (est. total)', `$${info.spend.toFixed(2)}`],
          ]} />
        </Card>
        <Card title="Parameters (frozen)">
          {info.params
            ? <KV items={Object.entries(info.params).map(([k, v]) => [k, String(v)])} />
            : <span style={{ color: '#556070' }}>unknown (pre-dashboard tag)</span>}
        </Card>
      </div>
      {info.has_task && (
        <Card title="Workers">
          <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
            <Button label="Start all" disabled={!anyStartable} onClick={() => act({ action: 'start' })} />
            <Button label="Pause all" disabled={!anyRunning} onClick={() => act({ action: 'pause' })} />
            <span title={anyRunning ? 'pause all workers before removing them' : undefined}>
              <Button
                label="Remove all" tone="danger"
                disabled={anyRunning || info.workers.length === 0}
                onClick={() => act({ action: 'remove' })}
              />
            </span>
          </div>
          <WorkersTable
            workers={info.workers}
            onAction={(workerId, action) => act({ worker_id: workerId, action })}
          />
          <AddWorkerForms workload={workload} tag={tag} onError={setError} onChanged={refresh} />
        </Card>
      )}
      {error && <div style={{ color: '#b23b3b', fontSize: 13 }}>{error}</div>}
    </>
  );
}

const fmt = (v: number | null | undefined, digits = 1, suffix = '') =>
  v == null ? '—' : `${v.toFixed(digits)}${suffix}`;

function StatsFigure({ tag, name, version }: { tag: string; name: string; version: number }) {
  const [item, setItem] = useState<unknown | null>(null);
  useEffect(() => {
    getJSON(`/api/kill_test/figure/${name}?tag=${encodeURIComponent(tag)}`)
      .then((d) => setItem(d.item ?? null))
      .catch(() => setItem(null));
  }, [tag, name, version]);
  if (!item) return null;
  return <div style={{ marginTop: 12 }}><BokehFigure item={item} /></div>;
}

// Stats-table columns: header label + how to read the sort key from a row.
const STAT_COLUMNS: [string, (w: WorkerStats) => string | number | null][] = [
  ['worker', (w) => w.worker_id],
  ['kind', (w) => w.kind],
  ['threads', (w) => w.threads],
  ['arch', (w) => w.host_arch],
  ['pairs', (w) => w.pairs_total],
  ['pairs/hr', (w) => w.pairs_per_hour],
  ['self-play s', (w) => w.gen_s],
  ['sim s', (w) => w.sim_s],
  ['upload s', (w) => w.upload_s],
  ['upload MB/s', (w) => w.upload_mbps],
  ['updated', (w) => w.updated_at],
];

function StatsTab({ tag }: { tag: string }) {
  const [workers, setWorkers] = useState<WorkerStats[]>([]);
  const [version, setVersion] = useState(0);
  const [sortCol, setSortCol] = useState(0);
  const [sortAsc, setSortAsc] = useState(true);
  const lastUpdate = useRef(0);

  useEffect(() => {
    let live = true;
    const refresh = async () => {
      try {
        const d = await getJSON(`/api/kill_test/stats?tag=${encodeURIComponent(tag)}`);
        if (!live) return;
        setWorkers(d.workers);
        if (d.updated_at !== lastUpdate.current) {
          lastUpdate.current = d.updated_at;
          setVersion((v) => v + 1); // stats advanced -> re-fetch the figures
        }
      } catch { /* API may be restarting; the poll retries */ }
    };
    refresh();
    const id = setInterval(refresh, 5000);
    return () => { live = false; clearInterval(id); };
  }, [tag]);

  if (workers.length === 0) {
    return (
      <div className="card" style={{ color: '#556070', fontStyle: 'italic' }}>
        No worker stats yet — they appear after each worker's first completed cycle.
      </div>
    );
  }

  const key = STAT_COLUMNS[sortCol][1];
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
    <div className="card">
      <table style={{ borderCollapse: 'collapse', fontSize: 13, width: '100%' }}>
        <thead>
          <tr style={{ textAlign: 'left', color: '#445063' }}>
            {STAT_COLUMNS.map(([h], i) => (
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
              <td style={{ padding: '5px 12px 5px 0', fontWeight: 600 }}>{w.worker_id}</td>
              <td style={{ padding: '5px 12px 5px 0' }}>{w.kind}</td>
              <td style={{ padding: '5px 12px 5px 0' }}>{w.threads ?? '—'}</td>
              <td style={{ padding: '5px 12px 5px 0' }}>
                {w.host_arch ?? '—'}
                {w.bundle_arch && w.bundle_arch !== w.host_arch ? ` (runs ${w.bundle_arch})` : ''}
              </td>
              <td style={{ padding: '5px 12px 5px 0' }}>{w.pairs_total}</td>
              <td style={{ padding: '5px 12px 5px 0' }}>{fmt(w.pairs_per_hour)}</td>
              <td style={{ padding: '5px 12px 5px 0' }}>{fmt(w.gen_s)}</td>
              <td style={{ padding: '5px 12px 5px 0' }}>{fmt(w.sim_s)}</td>
              <td style={{ padding: '5px 12px 5px 0' }}>{fmt(w.upload_s)}</td>
              <td style={{ padding: '5px 12px 5px 0' }}>{fmt(w.upload_mbps)}</td>
              <td style={{ padding: '5px 12px 5px 0' }}>{relTime(w.updated_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <StatsFigure tag={tag} name="pairs_timeline" version={version} />
      <StatsFigure tag={tag} name="throughput" version={version} />
      <StatsFigure tag={tag} name="cycle_breakdown" version={version} />
    </div>
  );
}

export default function TaskView({ workload, tag }: { workload: Workload; tag: string }) {
  const tabs = workload.name === 'kill_test' ? ['Overview', 'Stats'] : ['Overview'];
  const [tab, setTab] = useState(0);
  return (
    <>
      <div style={{ borderBottom: '2px solid #1f77b4', marginBottom: 12, fontSize: 14, display: 'flex', gap: 2 }}>
        {tabs.map((name, i) => (
          <span
            key={name}
            onClick={() => setTab(i)}
            style={{
              padding: '5px 14px', cursor: 'pointer', borderRadius: '6px 6px 0 0', fontSize: 14,
              background: i === tab ? '#1f77b4' : '#dde6ef', color: i === tab ? 'white' : '#2c3540',
            }}
          >
            {name}
          </span>
        ))}
      </div>
      {tabs[tab] === 'Stats' ? <StatsTab tag={tag} /> : <OverviewTab workload={workload} tag={tag} />}
    </>
  );
}
