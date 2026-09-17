import { useContext, Component, ReactNode, useCallback, useEffect, useState } from 'react';
import { getJSON, postJSON } from '../../lib/api';
import { WORKLOAD_TABS } from '../../workloads';
import { Button, Role, Workload } from './MasterApp';
import StatsTab from './StatsTab';
import { TabActiveContext } from '../TabActiveContext';

// One task's view in the master dashboard: an Overview tab (frozen params,
// progress counters, the worker slots with per-role add/pause/start/remove
// controls), a generic Stats tab whenever the workload's roles publish stats,
// and the workload's own tabs from the client registry (web/src/workloads.tsx)
// -- e.g. the training workloads' Loss/Positions/Controls views.

type WorkerInfo = {
  worker_id: string; role: string; kind: 'local' | 'ssh'; desired_state: string; state: string;
  observed_running: boolean;
  threads: number | null; host: string | null; ssh?: string;
  gate_reason?: string; bundle_id: string | null; exit_reason?: string; retry_in_s?: number;
  undelivered: number | null; launched: boolean;
  // ssh: the task's machine the slot runs on (null for a bare host string).
  machine: string | null;
};

// One of the task's machines (registered by the operator, or rented for the
// task): where its ssh slots run. `state` is the reconcile pass's probe.
type MachineInfo = {
  name: string; provider: string; host: string; gpu_count: number | null;
  instance_type: string | null; instance_id: string | null; spot: boolean;
  cost_per_hr: number | null; spend: number;
  state: string; slots: string[]; exit_reason?: string; retry_in_s?: number;
};

// The provider, the account it rents as, and its catalog (GET /api/cloud/rental_offer).
type MachineType = {
  id: string; vcpus: number; gpu_count: number; gpu: string; arch: string; cost_per_hr: number;
};
type RentalOffer = {
  provider: string; account: string; types: MachineType[];
  spot_prices: Record<string, number>;  // current spot rate by type id, where readable
};

// An instance the provider tagged ours that no task names (GET /api/cloud/orphans).
type Orphan = { instance_id: string; type_id: string; state: string; owner: string | null; uptime_s: number | null };

// A slot still on the bundle the task has moved off. It joins the task's bundle
// by being replaced, which reconcile does once the slot is stopped and has
// handed over everything it collected -- until then it simply runs on.
function onOldBundle(w: WorkerInfo, taskBundle: string | null): boolean {
  return w.bundle_id != null && taskBundle != null && w.bundle_id !== taskBundle;
}

// A short amber aside after a worker's resources: something true of the slot
// that is not its state.
function Note({ text, title }: { text: string; title: string }) {
  return <span style={{ color: '#a05a00' }} title={title}>{' '}· {text}</span>;
}

function workerResources(w: WorkerInfo): string {
  if (w.kind === 'local') return `${w.threads} threads`;
  return `${w.machine ?? w.host}${w.threads ? ` (${w.threads} threads)` : ''}`;
}

// Removing an ssh slot deletes its container, and with it any finished output
// the collection pass has not moved yet. Reconcile's own replace path waits for
// a collection to report zero before it does that; an operator is told what
// they would be discarding instead of being stopped, since discarding it is
// sometimes exactly the intent.
function discardWarning(workers: WorkerInfo[]): string | null {
  // A slot whose container was never created holds nothing by definition;
  // warning about it would spend the dialog's credibility on a false alarm.
  const holding = workers.filter((w) => w.kind === 'ssh' && w.launched && w.undelivered !== 0);
  if (holding.length === 0) return null;
  const described = holding.map((w) => (
    w.undelivered == null
      ? `${w.worker_id} (never collected from)`
      : `${w.worker_id} (${w.undelivered} files)`
  ));
  return `Discard finished output still held by ${described.join(', ')}?`;
}

// A slot mid-transition: its process/container has not yet caught up to the operator's
// intent, so its Start/Pause control is disabled and shows a spinner.
const IN_FLIGHT = new Set(['starting', 'stopping']);
// Mirrors IDLE_STOP_SECONDS in py/scribblez/dashboard/workers.py.
const IDLE_STOP_MINUTES = 10;
type ProfileChange = { name: string; profile: number | boolean | string; task: number | boolean | string };
type TaskInfo = {
  workload: string; tag: string; has_task: boolean; params: Record<string, number | boolean | string> | null;
  // The parameter profile the params were resolved from ('' if none), and how
  // the frozen params depart from it -- provenance, not live configuration.
  profile: string; profile_diff: ProfileChange[];
  created_at: number | null; progress: [string, string | number][]; gates: Record<string, string>;
  data_dir: string; workers: WorkerInfo[]; machines: MachineInfo[]; spend: number;
  bundle_id: string | null; bundle_drift: boolean;
};

const stateColors: Record<string, string> = {
  running: '#2a7a2a', paused: '#8494a5', exited: '#b23b3b', finished: '#446e9b',
  up: '#2a7a2a', 'no docker': '#b23b3b', launching: '#1f77b4', preparing: '#1f77b4',
  stopped: '#8494a5', gone: '#b23b3b',
  waiting: '#a05a00',
  unreachable: '#a05a00',
  starting: '#1f77b4', stopping: '#1f77b4',
  // No reconcile pass has observed this slot yet (only that pass talks to the
  // machines; a status request reads what it left behind).
  checking: '#8494a5',
};

type Busy = 'local' | 'ssh' | null;
type AddWorker = (kind: 'local' | 'ssh', body: Record<string, unknown>) => void;

// Contains a render error to its tab: a crashing tab shows an inline message
// instead of unmounting (blanking) the whole dashboard. Tabs stay mounted when
// hidden, so instead of resetting by remount, the error clears when the tab is
// re-activated -- navigating away and back retries the tab.
class TabErrorBoundary extends Component<
  { active: boolean; children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidUpdate(prev: { active: boolean }) {
    if (this.state.error && this.props.active && !prev.active) this.setState({ error: null });
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

const helpText = { fontSize: 12, color: '#556070', marginTop: 6 };
// The local (threads) add-worker form.
function LocalForm({ add, busy, disabled }: { add: AddWorker; busy: Busy; disabled: boolean }) {
  const [threads, setThreads] = useState('');
  return (
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
  );
}

// The ssh add-worker form: one of the task's machines, or a bare SSH
// destination ("user@host" or an ~/.ssh/config alias, passed to ssh verbatim),
// plus an optional thread count. Machine prerequisites: docs/master_dashboard.md.
function SshForm({ machines, add, busy, disabled }: {
  machines: MachineInfo[]; add: AddWorker; busy: Busy; disabled: boolean;
}) {
  // '' selects the free host field; otherwise a machine name.
  const [machine, setMachine] = useState(machines[0]?.name ?? '');
  const [host, setHost] = useState('');
  const [threads, setThreads] = useState('');
  const target = machines.some((m) => m.name === machine) ? machine : '';
  const ready = target ? true : host.trim() !== '';
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
      <label style={{ fontSize: 13 }}>
        SSH — machine<br />
        <select
          style={{ ...numInput, width: 160 }} value={target}
          onChange={(e) => setMachine(e.target.value)}
        >
          {machines.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
          <option value="">a host by address…</option>
        </select>
      </label>
      {!target && (
        <label style={{ fontSize: 13 }}>
          host<br />
          <input
            style={{ ...numInput, width: 160 }} value={host} placeholder="user@host"
            onChange={(e) => setHost(e.target.value)}
          />
        </label>
      )}
      <label style={{ fontSize: 13 }}>
        threads<br />
        <input
          style={numInput} value={threads} placeholder="all"
          onChange={(e) => setThreads(e.target.value)}
        />
      </label>
      <Button
        label={busy === 'ssh' ? 'Adding…' : 'Add ssh'}
        disabled={disabled || !ready}
        onClick={() => add('ssh', {
          ...(target ? { machine: target } : { host: host.trim() }),
          threads: threads ? parseInt(threads, 10) : null,
        })}
      />
    </div>
  );
}

// The task's machines: what its ssh slots run on. Registering one records
// its address and key; the reconcile pass probes it (ssh + Docker) like it
// probes the slots. Removing a machine removes the slots on it, under the
// slot rule (nothing running, nothing unreachable, output discards confirmed).
// The rental offer, fetched once per page load and shared by every task's
// rent form; a failed fetch (no aws credentials yet) leaves the form out
// rather than broken, with the reason shown once.
let offerPromise: Promise<RentalOffer> | null = null;
function loadRentalOffer(): Promise<RentalOffer> {
  if (!offerPromise) {
    offerPromise = getJSON('/api/cloud/rental_offer').catch((e) => {
      offerPromise = null;
      throw e;
    });
  }
  return offerPromise;
}

// Renting: pick a type from the catalog; the machine appears as `launching`
// and reads `up` once its first-boot script has pulled the worker images.
// The name is an alias for the tables and the instance's ownership tag;
// left empty, the server picks one.
function RentForm({ offer, busy, onRent }: {
  offer: RentalOffer; busy: boolean; onRent: (name: string, typeId: string, spot: boolean) => void;
}) {
  const types = offer.types;
  const [name, setName] = useState('');
  const [typeId, setTypeId] = useState(types[0]?.id ?? '');
  const [spot, setSpot] = useState(false);
  const t = types.find((x) => x.id === typeId) ?? types[0];
  const spotOf = (x: MachineType) => offer.spot_prices[x.id];
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 13, fontWeight: 600 }}>
        Rent on {offer.provider.toUpperCase()}
        <span style={{ fontWeight: 400, color: '#556070' }}> — {offer.account}</span>
      </div>
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap', marginTop: 4 }}>
        <label style={{ fontSize: 13 }}>
          name (optional)<br />
          <input
            style={{ ...numInput, width: 120 }} value={name} placeholder={`${offer.provider}-1`}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <label style={{ fontSize: 13 }}>
          type<br />
          <select style={{ ...numInput, width: 'auto', minWidth: 300 }} value={t?.id ?? ''} onChange={(e) => setTypeId(e.target.value)}>
            {types.map((x) => (
              <option key={x.id} value={x.id}>
                {x.id} — {x.vcpus} vCPU{x.gpu ? `, ${x.gpu_count}× ${x.gpu}` : ''} — ${x.cost_per_hr.toFixed(3)}/hr
                {spotOf(x) != null ? ` (spot $${spotOf(x).toFixed(3)})` : ''}
              </option>
            ))}
          </select>
        </label>
        <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 4, paddingBottom: 6 }}
          title="spare capacity at its market rate; AWS may stop the machine when it wants the capacity back, and starts it again when it is free (a trainer resumes from its checkpoint)">
          <input type="checkbox" checked={spot} onChange={(e) => setSpot(e.target.checked)} />
          spot{t && spotOf(t) != null ? ` ($${spotOf(t).toFixed(3)}/hr now)` : ''}
        </label>
        <Button
          label={busy ? 'Working…' : 'Rent'} disabled={busy || !t}
          onClick={() => onRent(name.trim(), t.id, spot)}
        />
      </div>
    </div>
  );
}

function MachinesCard({ workload, tag, machines, workers, onError, onChanged }: {
  workload: Workload; tag: string; machines: MachineInfo[]; workers: WorkerInfo[];
  onError: (e: string) => void; onChanged: () => void;
}) {
  const tabActive = useContext(TabActiveContext);
  const [name, setName] = useState('');
  const [host, setHost] = useState('');
  const [key, setKey] = useState('');
  const [gpus, setGpus] = useState('');
  const [busy, setBusy] = useState(false);
  const [offer, setOffer] = useState<RentalOffer | null>(null);
  const [offerError, setOfferError] = useState('');
  const [orphans, setOrphans] = useState<Orphan[]>([]);
  useEffect(() => {
    let alive = true;
    loadRentalOffer()
      .then((o) => { if (alive) setOffer(o); })
      .catch((e) => { if (alive) setOfferError(String(e)); });
    return () => { alive = false; };
  }, []);
  useEffect(() => {
    if (!tabActive || !offer) return;
    const poll = () => getJSON('/api/cloud/orphans').then((d) => setOrphans(d.orphans)).catch(() => {});
    poll();
    const id = setInterval(poll, 15000);
    return () => clearInterval(id);
  }, [tabActive, offer]);
  const post = async (path: string, body: Record<string, unknown>) => {
    setBusy(true);
    onError('');
    try {
      await postJSON(path, { workload: workload.name, tag, ...body });
      onChanged();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  };
  const remove = (m: MachineInfo) => {
    const warning = discardWarning(workers.filter((w) => w.machine === m.name));
    if (warning && !window.confirm(warning)) return;
    if (m.instance_id && m.state !== 'gone'
      && !window.confirm(`Terminate ${m.name} (${m.instance_type})? Its disk goes with it.`)) return;
    post('/api/task/machine_action', { name: m.name, action: 'remove' });
  };
  const terminateOrphan = (o: Orphan) => {
    if (!window.confirm(`Terminate ${o.instance_id} (${o.type_id})? No task tracks it.`)) return;
    post('/api/cloud/orphan_action', { instance_id: o.instance_id, action: 'terminate' });
  };
  return (
    <Card title="Machines">
      {machines.length > 0 && (
        <table style={{ borderCollapse: 'collapse', fontSize: 14, width: '100%', marginBottom: 10 }}>
          <thead>
            <tr style={{ textAlign: 'left', color: '#445063' }}>
              {['machine', 'host', 'provider', 'GPUs', 'state', 'slots', '$/hr', 'spend', ''].map((h) => (
                <th key={h} style={{ padding: '4px 14px 4px 0' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {machines.map((m) => {
              const busySlots = workers.some((w) => w.machine === m.name && (w.observed_running || IN_FLIGHT.has(w.state)));
              return (
                <tr key={m.name} style={{ borderTop: '1px solid #e2e8ee' }}>
                  <td style={{ padding: '6px 14px 6px 0', fontWeight: 600 }}>{m.name}</td>
                  <td style={{ padding: '6px 14px 6px 0', fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>{m.host}</td>
                  <td style={{ padding: '6px 14px 6px 0' }}>{m.instance_type ? `${m.provider} ${m.instance_type}${m.spot ? ' spot' : ''}` : m.provider}</td>
                  <td style={{ padding: '6px 14px 6px 0' }}>{m.gpu_count ?? '?'}</td>
                  <td style={{ padding: '6px 14px 6px 0', color: stateColors[m.state] ?? '#1a1f28', fontWeight: 600 }}>
                    {m.state}
                    {m.exit_reason && (
                      <div style={{ fontWeight: 400, fontSize: 12, color: '#a05a00', maxWidth: 460 }} title={m.exit_reason}>
                        {m.exit_reason}
                        {m.retry_in_s != null && (
                          <span style={{ color: '#6b7280' }}>
                            {' '}{m.retry_in_s > 0 ? `Next attempt in ${m.retry_in_s} s.` : 'Retrying now.'}
                          </span>
                        )}
                      </div>
                    )}
                  </td>
                  <td style={{ padding: '6px 14px 6px 0' }}>{m.slots.length ? m.slots.join(', ') : '—'}</td>
                  <td style={{ padding: '6px 14px 6px 0' }}>{m.cost_per_hr != null ? `$${m.cost_per_hr}` : '—'}</td>
                  <td style={{ padding: '6px 14px 6px 0' }}>{m.instance_id ? `$${m.spend.toFixed(2)}` : '—'}</td>
                  <td style={{ padding: '6px 0' }}>
                    <span title={busySlots ? 'pause the slots on it before removing the machine' : undefined}>
                      <Button label="Remove" tone="danger" disabled={busy || busySlots} onClick={() => remove(m)} />
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <label style={{ fontSize: 13 }}>
          Register — name<br />
          <input style={{ ...numInput, width: 120 }} value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label style={{ fontSize: 13 }}>
          host<br />
          <input style={{ ...numInput, width: 160 }} value={host} placeholder="user@host" onChange={(e) => setHost(e.target.value)} />
        </label>
        <label style={{ fontSize: 13 }}>
          key file (optional)<br />
          <input style={{ ...numInput, width: 220 }} value={key} placeholder="container's own identity" onChange={(e) => setKey(e.target.value)} />
        </label>
        <label style={{ fontSize: 13 }}>
          GPUs<br />
          <input style={numInput} value={gpus} placeholder="unknown" onChange={(e) => setGpus(e.target.value)} />
        </label>
        <Button
          label={busy ? 'Working…' : 'Register'}
          disabled={busy || !name.trim() || !host.trim()}
          onClick={() => post('/api/task/machines', {
            name: name.trim(), host: host.trim(), identity_file: key.trim() || null,
            gpu_count: gpus.trim() ? parseInt(gpus, 10) : null,
          })}
        />
      </div>
      <div style={helpText}>
        a machine you prepared (ssh key, Docker, the worker image pulled): docs/master_dashboard.md.
      </div>
      {offer && offer.types.length > 0 && (
        <RentForm
          offer={offer} busy={busy}
          onRent={(n, typeId, spot) => post('/api/task/machines', { name: n, type_id: typeId, spot })}
        />
      )}
      {offer && (
        <div style={helpText}>
          a rented machine is stopped after {IDLE_STOP_MINUTES} idle minutes (disk kept, no hourly charge)
          and started again when a slot on it is started; Remove terminates it.
        </div>
      )}
      {offerError && <div style={{ ...helpText, color: '#a05a00' }}>renting unavailable: {offerError}</div>}
      {orphans.length > 0 && offer && (
        <div style={{ marginTop: 10, color: '#a05a00', fontSize: 13 }}>
          <b>{offer.provider.toUpperCase()} instances tagged ours that no task tracks</b> (billing until terminated):
          <table style={{ borderCollapse: 'collapse', fontSize: 13, marginTop: 4 }}>
            <tbody>
              {orphans.map((o) => (
                <tr key={o.instance_id}>
                  <td style={{ padding: '3px 14px 3px 0', fontFamily: 'ui-monospace, monospace' }}>{o.instance_id}</td>
                  <td style={{ padding: '3px 14px 3px 0' }}>{o.type_id}</td>
                  <td style={{ padding: '3px 14px 3px 0' }}>{o.state}</td>
                  <td style={{ padding: '3px 14px 3px 0' }}>{o.owner ?? 'no owner tag'}</td>
                  <td style={{ padding: '3px 14px 3px 0' }}>{o.uptime_s != null ? `${Math.round(o.uptime_s / 60)} min` : '—'}</td>
                  <td><Button label="Terminate" tone="danger" disabled={busy} onClick={() => terminateOrphan(o)} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

// The add-worker forms for one role: a local form (threads) and/or an ssh
// form (a machine), per the role's declared kinds. A singleton role's forms
// disable once it has a slot. Adding only records a paused slot; nothing
// launches until the operator starts it from the workers table.
function AddWorkerForms({ workload, role, tag, taken, machines, onError, onChanged }: {
  workload: Workload; role: Role; tag: string; taken: boolean; machines: MachineInfo[];
  onError: (e: string) => void; onChanged: () => void;
}) {
  // Which form is mid-request ('local' | 'ssh' | null): its button
  // shows a progress label.
  const [busy, setBusy] = useState<Busy>(null);
  const disabled = busy !== null || (role.singleton && taken);

  const add: AddWorker = async (kind, body) => {
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
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 12, alignItems: 'flex-start' }}>
      <span
        style={{ fontSize: 13, fontWeight: 600, width: 210, flexShrink: 0, marginTop: 4 }}
        title={role.singleton ? 'at most one worker' : undefined}
      >
        {role.title}{role.singleton ? ' (singleton)' : ''}
      </span>
      {role.kinds.includes('local') && <LocalForm add={add} busy={busy} disabled={disabled} />}
      {role.kinds.includes('ssh') && <SshForm machines={machines} add={add} busy={busy} disabled={disabled} />}
    </div>
  );
}

function WorkersTable({ workers, taskBundle, onAction }: {
  workers: WorkerInfo[]; taskBundle: string | null;
  onAction: (workerId: string, action: string) => void;
}) {
  const onRemove = (w: WorkerInfo) => {
    const warning = discardWarning([w]);
    if (warning && !window.confirm(warning)) return;
    onAction(w.worker_id, 'remove');
  };
  if (workers.length === 0) {
    return <div style={{ color: '#556070', fontStyle: 'italic' }}>No workers yet.</div>;
  }
  return (
    <table style={{ borderCollapse: 'collapse', fontSize: 14, width: '100%' }}>
      <thead>
        <tr style={{ textAlign: 'left', color: '#445063' }}>
          {['worker', 'role', 'kind', 'resources', 'state', 'connect', ''].map((h) => (
            <th key={h} style={{ padding: '4px 14px 4px 0' }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {workers.map((w) => {
          const inFlight = IN_FLIGHT.has(w.state);
          // The toggle reflects operator intent, not the momentary state: a
          // still-winding-down worker keeps showing Pause until it has
          // actually stopped, never a misleading Start. Pause is never
          // disabled — it is always a safe intent write, and for a cloud slot
          // whose container creation keeps failing (stuck `starting`) it is the only
          // way back to a removable state.
          const desiredRunning = w.desired_state === 'running';
          return (
            <tr key={w.worker_id} style={{ borderTop: '1px solid #e2e8ee' }}>
              <td style={{ padding: '6px 14px 6px 0', fontWeight: 600 }}>{w.worker_id}</td>
              <td style={{ padding: '6px 14px 6px 0' }}>{w.role}</td>
              <td style={{ padding: '6px 14px 6px 0' }}>{w.kind}</td>
              <td style={{ padding: '6px 14px 6px 0' }}>
                {workerResources(w)}
                {w.undelivered ? (
                  <Note
                    text={`${w.undelivered} pending`}
                    title="finished output the container still holds; collected a batch per pass"
                  />
                ) : null}
                {onOldBundle(w, taskBundle) && (
                  <Note text="old bundle" title={`created on bundle ${w.bundle_id}`} />
                )}
              </td>
              <td
                style={{ padding: '6px 14px 6px 0', color: stateColors[w.state] ?? '#1a1f28', fontWeight: 600 }}
                title={w.gate_reason ? `parked by the scheduler: ${w.gate_reason}` : undefined}
              >
                {w.state}{w.state === 'waiting' && w.gate_reason ? ` (${w.gate_reason})` : ''}
                {/* Why it is not running, from the container itself: a worker
                    that cannot start says so here rather than only in its log. */}
                {w.exit_reason && (
                  <div
                    style={{ fontWeight: 400, fontSize: 12, color: '#a05a00', maxWidth: 460 }}
                    title={w.exit_reason}
                  >
                    {w.exit_reason}
                    {/* A slot in a long backoff should not read as one nobody
                        is retrying: say when. */}
                    {w.retry_in_s != null && (
                      <span style={{ color: '#6b7280' }}>
                        {' '}{w.retry_in_s > 0 ? `Next attempt in ${w.retry_in_s} s.` : 'Retrying now.'}
                      </span>
                    )}
                  </div>
                )}
              </td>
              <td style={{ padding: '6px 14px 6px 0', fontFamily: 'ui-monospace, monospace', fontSize: 12 }}>
                {w.ssh ?? '—'}
              </td>
              <td style={{ padding: '6px 0', whiteSpace: 'nowrap', display: 'flex', gap: 6, alignItems: 'center' }}>
                {desiredRunning
                  ? <Button label="Pause" onClick={() => onAction(w.worker_id, 'pause')} />
                  : <Button label="Start" disabled={inFlight} onClick={() => onAction(w.worker_id, 'start')} />}
                {inFlight && <span className="scz-spinner" title={w.state} aria-label={w.state} />}
                <span title={w.observed_running ? 'pause the worker before removing it' : undefined}>
                  <Button
                    label="Remove" tone="danger" disabled={w.observed_running || inFlight}
                    onClick={() => onRemove(w)}
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
  const tabActive = useContext(TabActiveContext);
  const [info, setInfo] = useState<TaskInfo | null>(null);
  const [error, setError] = useState('');
  // A redeploy builds every arch before it pushes; on a cold tree that is
  // minutes, so the button reports itself rather than looking dead.
  const [deploying, setDeploying] = useState(false);

  const refresh = useCallback(() => {
    getJSON(`/api/task?workload=${workload.name}&tag=${encodeURIComponent(tag)}`)
      .then(setInfo)
      .catch((e) => setError(String(e)));
  }, [workload, tag]);
  useEffect(() => {
    if (!tabActive) return;
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [refresh, tabActive]);

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
  const redeploy = async () => {
    setError('');
    setDeploying(true);
    try {
      await postJSON('/api/task/deploy', { workload: workload.name, tag });
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setDeploying(false);
    }
  };
  const removeAll = () => {
    const warning = discardWarning(info.workers);
    if (warning && !window.confirm(warning)) return;
    act({ action: 'remove' });
  };
  // What the task's rented machines bill right now: those launching or up.
  const BILLING = new Set(['launching', 'preparing', 'up', 'unreachable', 'no docker']);
  const cloudCost = info.machines.reduce((s, m) => s + (BILLING.has(m.state) ? m.cost_per_hr ?? 0 : 0), 0);
  const anyStartable = info.workers.some((w) => w.desired_state !== 'running');
  const anyPausable = info.workers.some((w) => w.desired_state === 'running');
  // A worker can only be removed once it is truly stopped (no live process/container).
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
            ['bundle', info.bundle_id
              ? <span title={info.bundle_id}>{info.bundle_id.split('-')[0]}</span>
              : 'none yet (deployed when the first remote worker starts)'],
            ['cloud burn rate', `$${cloudCost.toFixed(3)}/hr`],
            ['cloud spend (est. total)', `$${info.spend.toFixed(2)}`],
          ]} />
        </Card>
        <Card title="Parameters (frozen)">
          {info.params && (
            <div style={{ fontSize: 13, color: '#556070', marginBottom: 8 }}>
              {info.profile ? <>profile <b>{info.profile}</b></> : 'no profile'}
              {info.profile_diff.length > 0
                ? <>, changed: {info.profile_diff.map((c) => (
                    <span key={c.name} style={{ marginLeft: 6 }}>
                      <b>{c.name}</b> {String(c.profile)} → {String(c.task)}
                    </span>
                  ))}</>
                : info.profile ? ', unchanged' : ''}
            </div>
          )}
          {info.params
            ? <KV items={Object.entries(info.params).map(([k, v]) => [k, String(v)])} />
            : <span style={{ color: '#556070' }}>unknown (pre-dashboard tag)</span>}
        </Card>
      </div>
      {info.bundle_drift && (
        <div className="card" style={{ color: '#a05a00', marginTop: 14, display: 'flex', gap: 12, alignItems: 'center' }}>
          <span>
            The controller's code has changed since this task pinned its bundle. Remote workers
            keep running the pinned one; redeploying builds the current tree, pushes it, and
            replaces them.
          </span>
          <Button
            label={deploying ? 'Deploying…' : 'Redeploy'} disabled={deploying}
            onClick={redeploy}
          />
        </div>
      )}
      {info.has_task && (
        <Card title="Workers">
          <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
            <Button label="Start all" disabled={!anyStartable} onClick={() => act({ action: 'start' })} />
            <Button label="Pause all" disabled={!anyPausable} onClick={() => act({ action: 'pause' })} />
            <span title={anyAlive ? 'pause all workers before removing them' : undefined}>
              <Button
                label="Remove all" tone="danger"
                disabled={anyAlive || info.workers.length === 0}
                onClick={() => removeAll()}
              />
            </span>
          </div>
          <WorkersTable
            workers={info.workers} taskBundle={info.bundle_id}
            onAction={(workerId, action) => act({ worker_id: workerId, action })}
          />
          {workload.roles.map((role) => (
            <AddWorkerForms
              key={role.name} workload={workload} role={role} tag={tag}
              taken={info.workers.some((w) => w.role === role.name)} machines={info.machines}
              onError={setError} onChanged={refresh}
            />
          ))}
        </Card>
      )}
      {info.has_task && workload.roles.some((r) => r.kinds.includes('ssh')) && (
        <MachinesCard
          workload={workload} tag={tag} machines={info.machines} workers={info.workers}
          onError={setError} onChanged={refresh}
        />
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
  // Tabs render lazily on first visit but then stay mounted behind display:none:
  // switching back is instant (embedded figures and fetched state survive), and
  // TabActiveContext lets a hidden tab pause its background polling.
  const [visited, setVisited] = useState<Set<string>>(() => new Set([tabs[0]]));

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
            onClick={() => {
              setTab(i);
              setVisited((v) => (v.has(name) ? v : new Set(v).add(name)));
            }}
            style={{
              padding: '5px 14px', cursor: 'pointer', borderRadius: '6px 6px 0 0', fontSize: 14,
              background: i === tab ? '#1f77b4' : '#dde6ef', color: i === tab ? 'white' : '#2c3540',
            }}
          >
            {name}
          </span>
        ))}
      </div>
      {tabs.filter((name) => visited.has(name)).map((name) => {
        const active = name === tabs[tab];
        return (
          <div key={name} style={{ display: active ? undefined : 'none' }}>
            <TabActiveContext.Provider value={active}>
              <TabErrorBoundary active={active}>{renderTab(name)}</TabErrorBoundary>
            </TabActiveContext.Provider>
          </div>
        );
      })}
    </>
  );
}
