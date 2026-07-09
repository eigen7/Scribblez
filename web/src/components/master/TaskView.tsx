import { Component, ReactNode, useCallback, useEffect, useState } from 'react';
import { getJSON, postJSON } from '../../lib/api';
import { WORKLOAD_TABS } from '../../workloads';
import { Button, Role, Workload } from './MasterApp';
import StatsTab from './StatsTab';

// One task's view in the master dashboard: an Overview tab (frozen params,
// progress counters, the worker slots with per-role add/pause/start/remove
// controls), a generic Stats tab whenever the workload's roles publish stats,
// and the workload's own tabs from the client registry (web/src/workloads.tsx)
// -- e.g. the training workloads' Loss/Positions/Controls views.

type WorkerInfo = {
  worker_id: string; role: string; kind: 'local' | 'cloud'; desired_state: string; state: string;
  observed_running: boolean;
  threads: number | null; vcpus: number | null; flavor: string | null;
  pod_id: string | null; cost_per_hr?: number; public_ip?: string; ssh?: string;
  gate_reason?: string;
};

// A slot mid-transition: its process/pod has not yet caught up to the operator's
// intent, so its Start/Pause control is disabled and shows a spinner.
const IN_FLIGHT = new Set(['starting', 'stopping']);
type TaskInfo = {
  workload: string; tag: string; has_task: boolean; params: Record<string, number | boolean | string> | null;
  created_at: number | null; progress: [string, string | number][]; gates: Record<string, string>;
  data_dir: string; workers: WorkerInfo[]; spend: number;
};

const stateColors: Record<string, string> = {
  running: '#2a7a2a', paused: '#8494a5', exited: '#b23b3b',
  interrupted: '#a05a00', terminated: '#b23b3b', waiting: '#a05a00',
  starting: '#1f77b4', stopping: '#1f77b4',
};

// The Runpod CPU flavor ids accepted by pod creation (the REST API's fixed
// enum -- there is no listing endpoint): generation 3/5, compute- / general- /
// memory-optimized. Live availability and pricing are only visible in the
// Runpod console.
const CPU_FLAVORS = ['cpu3c', 'cpu3g', 'cpu3m', 'cpu5c', 'cpu5g', 'cpu5m'];
const RUNPOD_CONSOLE = 'https://console.runpod.io/deploy';

// Contains a render error to the active tab: a crashing tab shows an inline
// message instead of unmounting (blanking) the whole dashboard. Reset by
// remounting on a `key` change (tab switch), so navigating away and back
// retries the tab.
class TabErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="card" style={{ color: '#a05a00', padding: 16 }}>
          <b>This tab hit an error.</b>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12, marginTop: 8 }}>
            {String(this.state.error.message || this.state.error)}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

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

// The add-worker forms for one role: a local form (threads) and/or a cloud
// form (count x vCPUs x flavor), per the role's declared kinds. A singleton
// role's forms disable once it has a slot.
function AddWorkerForms({ workload, role, tag, taken, onError, onChanged }: {
  workload: Workload; role: Role; tag: string; taken: boolean;
  onError: (e: string) => void; onChanged: () => void;
}) {
  const [threads, setThreads] = useState('');
  const [count, setCount] = useState('1');
  const [vcpus, setVcpus] = useState('16');
  const [flavor, setFlavor] = useState('cpu3c');
  // Which form is mid-request ('local' | 'cloud' | null): its button shows a
  // progress label -- pod creation takes a few seconds.
  const [busy, setBusy] = useState<'local' | 'cloud' | null>(null);
  const disabled = busy !== null || (role.singleton && taken);

  const add = async (kind: 'local' | 'cloud', body: Record<string, unknown>) => {
    setBusy(kind);
    onError('');
    try {
      await postJSON('/api/task/workers', {
        workload: workload.name, tag, kind, role: role.name, ...body,
      });
      onChanged();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div style={{ display: 'flex', gap: 40, flexWrap: 'wrap', marginTop: 12, alignItems: 'flex-end' }}>
      <span style={{ fontSize: 13, fontWeight: 600, minWidth: 90 }} title={role.singleton ? 'at most one worker' : undefined}>
        {role.title}{role.singleton ? ' (singleton)' : ''}
      </span>
      {role.kinds.includes('local') && (
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
          <label style={{ fontSize: 13 }}>
            Local — threads<br />
            <input
              style={numInput} value={threads} placeholder="all"
              onChange={(e) => setThreads(e.target.value)}
            />
          </label>
          <Button
            label={busy === 'local' ? 'Adding…' : 'Add local'}
            disabled={disabled}
            onClick={() => add('local', { threads: threads ? parseInt(threads, 10) : null })}
          />
        </div>
      )}
      {role.kinds.includes('cloud') && (
        <div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
            <label style={{ fontSize: 13 }}>
              Cloud — count<br />
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
              disabled={disabled}
              onClick={() => add('cloud', {
                count: parseInt(count, 10) || 1, vcpus: parseInt(vcpus, 10) || 16, flavor,
              })}
            />
          </div>
          <div style={{ fontSize: 12, color: '#556070', marginTop: 6 }}>
            flavors: cpu3/cpu5 = hardware generation; c/g/m = compute/general/memory-optimized.{' '}
            <a href={RUNPOD_CONSOLE} target="_blank" rel="noreferrer">availability &amp; pricing ↗</a>
            {role.interruptible &&
              ' — rented interruptible (discounted; auto-restarted if reclaimed)'}
          </div>
        </div>
      )}
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
          {['worker', 'role', 'kind', 'resources', 'state', '$/hr', 'connect', ''].map((h) => (
            <th key={h} style={{ padding: '4px 14px 4px 0' }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {workers.map((w) => {
          const inFlight = IN_FLIGHT.has(w.state);
          // The toggle reflects operator intent, not the momentary state: a
          // still-winding-down worker keeps showing Pause (disabled) until it
          // has actually stopped, never a misleading Start.
          const desiredRunning = w.desired_state === 'running';
          return (
            <tr key={w.worker_id} style={{ borderTop: '1px solid #e2e8ee' }}>
              <td style={{ padding: '6px 14px 6px 0', fontWeight: 600 }}>{w.worker_id}</td>
              <td style={{ padding: '6px 14px 6px 0' }}>{w.role}</td>
              <td style={{ padding: '6px 14px 6px 0' }}>{w.kind}</td>
              <td style={{ padding: '6px 14px 6px 0' }}>
                {w.kind === 'local' ? `${w.threads} threads` : `${w.vcpus} vcpu ${w.flavor}`}
              </td>
              <td
                style={{ padding: '6px 14px 6px 0', color: stateColors[w.state] ?? '#1a1f28', fontWeight: 600 }}
                title={w.gate_reason ? `parked by the scheduler: ${w.gate_reason}` : undefined}
              >
                {w.state}{w.state === 'waiting' && w.gate_reason ? ` (${w.gate_reason})` : ''}
              </td>
              <td style={{ padding: '6px 14px 6px 0' }}>
                {w.cost_per_hr != null ? `$${w.cost_per_hr}` : '—'}
              </td>
              <td style={{ padding: '6px 14px 6px 0', fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                {w.ssh ?? '—'}
              </td>
              <td style={{ padding: '6px 0', whiteSpace: 'nowrap', display: 'flex', gap: 6, alignItems: 'center' }}>
                {desiredRunning
                  ? <Button label="Pause" disabled={inFlight} onClick={() => onAction(w.worker_id, 'pause')} />
                  : <Button label="Start" disabled={inFlight} onClick={() => onAction(w.worker_id, 'start')} />}
                {inFlight && <span className="scz-spinner" title={w.state} aria-label={w.state} />}
                <span title={w.observed_running ? 'pause the worker before removing it' : undefined}>
                  <Button
                    label="Remove" tone="danger" disabled={w.observed_running || inFlight}
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
  // Cost accrues while a pod is really up (observed), not merely desired-running.
  const cloudCost = info.workers.reduce((s, w) => s + (w.observed_running ? w.cost_per_hr ?? 0 : 0), 0);
  const anyStartable = info.workers.some((w) => w.desired_state !== 'running');
  const anyPausable = info.workers.some((w) => w.desired_state === 'running');
  // A worker can only be removed once it is truly stopped (no live process/pod).
  const anyAlive = info.workers.some((w) => w.observed_running || IN_FLIGHT.has(w.state));

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
            ...info.progress.map(([k, v]): [string, React.ReactNode] => [k, String(v)]),
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
            <Button label="Pause all" disabled={!anyPausable} onClick={() => act({ action: 'pause' })} />
            <span title={anyAlive ? 'pause all workers before removing them' : undefined}>
              <Button
                label="Remove all" tone="danger"
                disabled={anyAlive || info.workers.length === 0}
                onClick={() => act({ action: 'remove' })}
              />
            </span>
          </div>
          <WorkersTable
            workers={info.workers}
            onAction={(workerId, action) => act({ worker_id: workerId, action })}
          />
          {workload.roles.map((role) => (
            <AddWorkerForms
              key={role.name} workload={workload} role={role} tag={tag}
              taken={info.workers.some((w) => w.role === role.name)}
              onError={setError} onChanged={refresh}
            />
          ))}
        </Card>
      )}
      {error && <div style={{ color: '#b23b3b', fontSize: 13 }}>{error}</div>}
    </>
  );
}

export default function TaskView({ workload, tag }: { workload: Workload; tag: string }) {
  const workloadTabs = WORKLOAD_TABS[workload.name] ?? [];
  const hasStats = workload.roles.some((r) => r.stats);
  const tabs = ['Overview', ...(hasStats ? ['Stats'] : []), ...workloadTabs.map((t) => t.name)];
  const [tab, setTab] = useState(0);

  const renderTab = (name: string): ReactNode => {
    if (name === 'Overview') return <OverviewTab workload={workload} tag={tag} />;
    if (name === 'Stats') return <StatsTab workload={workload.name} tag={tag} />;
    return workloadTabs.find((t) => t.name === name)?.render(workload.name, tag);
  };

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
      <TabErrorBoundary key={tabs[tab]}>{renderTab(tabs[tab])}</TabErrorBoundary>
    </>
  );
}
