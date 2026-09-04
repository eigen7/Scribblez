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
  worker_id: string; role: string; kind: 'local' | 'cloud' | 'ssh'; desired_state: string; state: string;
  observed_running: boolean;
  threads: number | null; vcpus: number | null; flavor: string | null;
  gpu_type_id: string | null; gpu_count: number | null;
  pod_id: string | null; host: string | null; cost_per_hr?: number; public_ip?: string; ssh?: string;
  gate_reason?: string; bundle_id: string | null; exit_reason?: string;
  undelivered: number | null; launched: boolean;
};

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
  if (w.kind === 'ssh') return `${w.host}${w.threads ? ` (${w.threads} threads)` : ''}`;
  if (w.gpu_type_id) return `${w.gpu_count}× ${w.gpu_type_id}`;
  return `${w.vcpus} vcpu ${w.flavor}`;
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

// A slot mid-transition: its process/pod has not yet caught up to the operator's
// intent, so its Start/Pause control is disabled and shows a spinner.
const IN_FLIGHT = new Set(['starting', 'stopping']);
type ProfileChange = { name: string; profile: number | boolean | string; task: number | boolean | string };
type TaskInfo = {
  workload: string; tag: string; has_task: boolean; params: Record<string, number | boolean | string> | null;
  // The parameter profile the params were resolved from ('' if none), and how
  // the frozen params depart from it -- provenance, not live configuration.
  profile: string; profile_diff: ProfileChange[];
  created_at: number | null; progress: [string, string | number][]; gates: Record<string, string>;
  data_dir: string; workers: WorkerInfo[]; spend: number;
  bundle_id: string | null; bundle_drift: boolean;
};

const stateColors: Record<string, string> = {
  running: '#2a7a2a', paused: '#8494a5', exited: '#b23b3b',
  interrupted: '#a05a00', terminated: '#b23b3b', waiting: '#a05a00',
  unreachable: '#a05a00',
  starting: '#1f77b4', stopping: '#1f77b4',
  // No reconcile pass has observed this slot yet (only that pass talks to the
  // machines; a status request reads what it left behind).
  checking: '#8494a5',
};

// The Runpod CPU flavor ids accepted by pod creation (the REST API's fixed
// enum -- there is no listing endpoint): generation 3/5, compute- / general- /
// memory-optimized. Used only as the fallback flavor list when live offers
// (/api/cloud/offers) are unreachable; otherwise the form shows live pricing.
const CPU_FLAVORS = ['cpu3c', 'cpu3g', 'cpu3m', 'cpu5c', 'cpu5g', 'cpu5m'];
const RUNPOD_CONSOLE = 'https://console.runpod.io/deploy';

// The live cloud instance catalog from GET /api/cloud/offers: CPU flavors and
// GPU types with current pricing and stock, feeding the add-cloud selector.
type CpuOffer = {
  id: string; display_name: string; group_name: string;
  min_vcpu: number; max_vcpu: number; ram_multiplier: number; disk_per_vcpu: number;
  price_per_vcpu_hr: number | null; stock: string | null;
};
type GpuOffer = {
  id: string; display_name: string; vram_gb: number;
  secure_price: number | null; community_price: number | null;
  secure_spot_price: number | null; community_spot_price: number | null;
  max_gpu_count: number; stock: string | null;
  min_vcpu: number | null; min_memory_gb: number | null;
  secure_available: boolean; community_available: boolean;
};
type CloudOffers = { cpu: CpuOffer[]; gpu: GpuOffer[] };

// Offers are fetched once per page load and shared across every role's cloud
// form (a module-level cached promise). A failed fetch clears the cache so a
// later form retries rather than being stuck on the fallback.
let offersPromise: Promise<CloudOffers> | null = null;
function loadOffers(): Promise<CloudOffers> {
  if (!offersPromise) {
    offersPromise = getJSON('/api/cloud/offers').catch((e) => {
      offersPromise = null;
      throw e;
    });
  }
  return offersPromise;
}

function useCloudOffers(): { offers: CloudOffers | null; failed: boolean } {
  const [offers, setOffers] = useState<CloudOffers | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let alive = true;
    loadOffers()
      .then((o) => { if (alive) setOffers(o); })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, []);
  return { offers, failed };
}

type Busy = 'local' | 'cloud' | 'ssh' | null;
type AddWorker = (kind: 'local' | 'cloud' | 'ssh', body: Record<string, unknown>) => void;

const STOCK_COLORS: Record<string, string> = { High: '#2a7a2a', Medium: '#a05a00', Low: '#b23b3b' };
function StockBadge({ stock }: { stock: string | null }) {
  const color = stock ? STOCK_COLORS[stock] ?? '#556070' : '#8494a5';
  return <span style={{ color, fontWeight: 600, fontSize: 12 }}>{stock ?? '—'}</span>;
}

const offerTable = { borderCollapse: 'collapse' as const, fontSize: 13 };
const offerTh = { textAlign: 'left' as const, color: '#445063', padding: '3px 12px 3px 0', fontWeight: 600 };
const offerTd = { padding: '3px 12px 3px 0', whiteSpace: 'nowrap' as const };
function offerRowStyle(selected: boolean) {
  return { cursor: 'pointer', background: selected ? '#dbe9f6' : undefined, borderTop: '1px solid #e2e8ee' };
}

const money = (v: number | null | undefined, digits = 3) => (v != null ? `$${v.toFixed(digits)}` : '—');
const clampInt = (raw: number, lo: number, hi: number) =>
  Number.isNaN(raw) ? lo : Math.min(hi, Math.max(lo, raw));

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
function ConsoleLink() {
  return <a href={RUNPOD_CONSOLE} target="_blank" rel="noreferrer">availability &amp; pricing ↗</a>;
}
function InterruptibleNote({ role }: { role: Role }) {
  if (!role.interruptible) return null;
  return <> — rented interruptible (discounted; auto-restarted if reclaimed)</>;
}

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

// The ssh (operator-owned machine) add-worker form: an SSH destination
// ("user@host" or an ~/.ssh/config alias, passed to ssh verbatim) plus an
// optional thread count. Machine prerequisites: docs/master_dashboard.md.
function SshForm({ add, busy, disabled }: { add: AddWorker; busy: Busy; disabled: boolean }) {
  const [host, setHost] = useState('');
  const [threads, setThreads] = useState('');
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
      <label style={{ fontSize: 13 }}>
        SSH — host<br />
        <input
          style={{ ...numInput, width: 160 }} value={host} placeholder="user@host"
          onChange={(e) => setHost(e.target.value)}
        />
      </label>
      <label style={{ fontSize: 13 }}>
        threads<br />
        <input
          style={numInput} value={threads} placeholder="all"
          onChange={(e) => setThreads(e.target.value)}
        />
      </label>
      <Button
        label={busy === 'ssh' ? 'Adding…' : 'Add ssh'}
        disabled={disabled || !host.trim()}
        onClick={() => add('ssh', {
          host: host.trim(), threads: threads ? parseInt(threads, 10) : null,
        })}
      />
    </div>
  );
}

// The CPU-flavor selector: a selectable table of live flavors (price, live RAM
// for the current vCPU input, disk, stock) plus count/vCPU inputs and a live
// per-pod cost estimate. The vCPU input is clamped to the selected flavor's
// min/max.
function CpuCloudForm({ role, offers, add, busy, disabled }: {
  role: Role; offers: CpuOffer[]; add: AddWorker; busy: Busy; disabled: boolean;
}) {
  const [count, setCount] = useState('1');
  const [vcpus, setVcpus] = useState('16');
  const [flavorId, setFlavorId] = useState(offers.some((f) => f.id === 'cpu3c') ? 'cpu3c' : offers[0].id);
  const flavor = offers.find((f) => f.id === flavorId) ?? offers[0];
  const rawV = parseInt(vcpus, 10);
  const vcpuN = clampInt(rawV, flavor.min_vcpu, flavor.max_vcpu);
  const n = parseInt(count, 10) || 1;
  const perPod = flavor.price_per_vcpu_hr != null ? flavor.price_per_vcpu_hr * vcpuN : null;

  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>Cloud — CPU flavor</div>
      <table style={offerTable}>
        <thead>
          <tr>{['', 'flavor', 'class', '$/vCPU/hr', 'RAM', 'disk/vCPU', 'stock'].map((h) => (
            <th key={h} style={offerTh}>{h}</th>
          ))}</tr>
        </thead>
        <tbody>
          {offers.map((f) => {
            const rowVcpu = clampInt(rawV, f.min_vcpu, f.max_vcpu);
            return (
              <tr key={f.id} onClick={() => setFlavorId(f.id)} style={offerRowStyle(f.id === flavorId)}>
                <td style={offerTd}><input type="radio" checked={f.id === flavorId} readOnly /></td>
                <td style={offerTd}>{f.id}</td>
                <td style={offerTd}>{f.group_name} {f.display_name}</td>
                <td style={offerTd}>{money(f.price_per_vcpu_hr)}</td>
                <td style={offerTd}>{f.ram_multiplier * rowVcpu} GB</td>
                <td style={offerTd}>{f.disk_per_vcpu} GB</td>
                <td style={offerTd}><StockBadge stock={f.stock} /></td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', marginTop: 8 }}>
        <label style={{ fontSize: 13 }}>
          count<br />
          <input style={numInput} value={count} onChange={(e) => setCount(e.target.value)} />
        </label>
        <label style={{ fontSize: 13 }}>
          vCPUs ({flavor.min_vcpu}–{flavor.max_vcpu})<br />
          <input style={numInput} value={vcpus} onChange={(e) => setVcpus(e.target.value)} />
        </label>
        <span style={{ fontSize: 13, color: '#2c3540', paddingBottom: 4 }}>
          {perPod != null
            ? `≈ ${money(perPod)}/hr per pod${n > 1 ? ` · ${money(perPod * n)}/hr for ${n}` : ''}`
            : 'price unavailable'}
        </span>
        <Button
          label={busy === 'cloud' ? 'Adding…' : 'Add cloud'}
          disabled={disabled}
          onClick={() => add('cloud', { count: n, vcpus: vcpuN, flavor: flavorId })}
        />
      </div>
      <div style={helpText}>
        <ConsoleLink /><InterruptibleNote role={role} />
      </div>
    </div>
  );
}

// The GPU-instance selector: a selectable table of available GPU types (VRAM,
// on-demand rate, spot rate when the role rents interruptible, max count,
// stock) plus count/GPU-count inputs and a live per-pod cost estimate. The
// GPU-count input is clamped to the type's max.
function GpuCloudForm({ role, offers, add, busy, disabled }: {
  role: Role; offers: GpuOffer[]; add: AddWorker; busy: Busy; disabled: boolean;
}) {
  const [count, setCount] = useState('1');
  const [gpuCount, setGpuCount] = useState('1');
  const [gpuId, setGpuId] = useState(offers[0].id);
  const gpu = offers.find((g) => g.id === gpuId) ?? offers[0];
  const gc = clampInt(parseInt(gpuCount, 10), 1, gpu.max_gpu_count);
  const n = parseInt(count, 10) || 1;
  const onDemand = (g: GpuOffer) => g.secure_price ?? g.community_price;
  const spot = (g: GpuOffer) => g.secure_spot_price ?? g.community_spot_price;
  // A pod is billed at the spot rate exactly when the role rents interruptible.
  const unit = role.interruptible ? (spot(gpu) ?? onDemand(gpu)) : onDemand(gpu);
  const perPod = unit != null ? unit * gc : null;
  const cols = ['', 'GPU', 'VRAM', 'on-demand $/hr', ...(role.interruptible ? ['spot $/hr'] : []), 'max', 'stock'];

  return (
    <div>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>Cloud — GPU instance</div>
      <table style={offerTable}>
        <thead>
          <tr>{cols.map((h) => <th key={h} style={offerTh}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {offers.map((g) => (
            <tr key={g.id} onClick={() => setGpuId(g.id)} style={offerRowStyle(g.id === gpuId)}>
              <td style={offerTd}><input type="radio" checked={g.id === gpuId} readOnly /></td>
              <td style={offerTd}>{g.display_name}</td>
              <td style={offerTd}>{g.vram_gb} GB</td>
              <td style={offerTd}>{money(onDemand(g), 2)}</td>
              {role.interruptible && <td style={offerTd}>{money(spot(g), 2)}</td>}
              <td style={offerTd}>{g.max_gpu_count}</td>
              <td style={offerTd}><StockBadge stock={g.stock} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', marginTop: 8 }}>
        <label style={{ fontSize: 13 }}>
          count<br />
          <input style={numInput} value={count} onChange={(e) => setCount(e.target.value)} />
        </label>
        <label style={{ fontSize: 13 }}>
          GPUs (1–{gpu.max_gpu_count})<br />
          <input style={numInput} value={gpuCount} onChange={(e) => setGpuCount(e.target.value)} />
        </label>
        <span style={{ fontSize: 13, color: '#2c3540', paddingBottom: 4 }}>
          {perPod != null
            ? `≈ ${money(perPod, 2)}/hr per pod${n > 1 ? ` · ${money(perPod * n, 2)}/hr for ${n}` : ''}`
            : 'price unavailable'}
        </span>
        <Button
          label={busy === 'cloud' ? 'Adding…' : 'Add cloud'}
          disabled={disabled}
          onClick={() => add('cloud', { count: n, gpu_type_id: gpuId, gpu_count: gc })}
        />
      </div>
      <div style={helpText}>
        <ConsoleLink /><InterruptibleNote role={role} />
      </div>
    </div>
  );
}

// Fallback when live offers are unreachable: the static CPU flavor enum in a
// plain dropdown, with the original help text.
function CpuFallbackForm({ role, add, busy, disabled }: {
  role: Role; add: AddWorker; busy: Busy; disabled: boolean;
}) {
  const [count, setCount] = useState('1');
  const [vcpus, setVcpus] = useState('16');
  const [flavor, setFlavor] = useState('cpu3c');
  return (
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
          label={busy === 'cloud' ? 'Adding…' : 'Add cloud'}
          disabled={disabled}
          onClick={() => add('cloud', {
            count: parseInt(count, 10) || 1, vcpus: parseInt(vcpus, 10) || 16, flavor,
          })}
        />
      </div>
      <div style={helpText}>
        flavors: cpu3/cpu5 = hardware generation; c/g/m = compute/general/memory-optimized.{' '}
        <ConsoleLink /><InterruptibleNote role={role} />
      </div>
    </div>
  );
}

// Fallback when live offers are unreachable for a GPU role: pricing is hidden
// and the operator names the Runpod gpuTypeId directly.
function GpuFallbackForm({ role, add, busy, disabled }: {
  role: Role; add: AddWorker; busy: Busy; disabled: boolean;
}) {
  const [count, setCount] = useState('1');
  const [gpuCount, setGpuCount] = useState('1');
  const [gpuId, setGpuId] = useState('');
  return (
    <div>
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
        <label style={{ fontSize: 13 }}>
          Cloud — count<br />
          <input style={numInput} value={count} onChange={(e) => setCount(e.target.value)} />
        </label>
        <label style={{ fontSize: 13 }}>
          GPU type id<br />
          <input
            style={{ ...numInput, width: 200 }} value={gpuId} placeholder="e.g. NVIDIA A100 80GB PCIe"
            onChange={(e) => setGpuId(e.target.value)}
          />
        </label>
        <label style={{ fontSize: 13 }}>
          GPUs<br />
          <input style={numInput} value={gpuCount} onChange={(e) => setGpuCount(e.target.value)} />
        </label>
        <Button
          label={busy === 'cloud' ? 'Adding…' : 'Add cloud'}
          disabled={disabled || !gpuId.trim()}
          onClick={() => add('cloud', {
            count: parseInt(count, 10) || 1,
            gpu_type_id: gpuId.trim(),
            gpu_count: parseInt(gpuCount, 10) || 1,
          })}
        />
      </div>
      <div style={helpText}>
        pricing unavailable — enter a Runpod GPU type id. <ConsoleLink /><InterruptibleNote role={role} />
      </div>
    </div>
  );
}

// The cloud add-worker form: live GPU or CPU selector per the role, with a
// static fallback when the offers fetch fails or returns nothing usable.
function CloudForm({ role, add, busy, disabled }: {
  role: Role; add: AddWorker; busy: Busy; disabled: boolean;
}) {
  const { offers, failed } = useCloudOffers();
  const usable = offers ? (role.gpu ? offers.gpu.length : offers.cpu.length) : 0;
  if (!offers && !failed) {
    return <div style={{ fontSize: 13, color: '#556070' }}>Loading instance offers…</div>;
  }
  if (!offers || !usable) {
    return role.gpu
      ? <GpuFallbackForm role={role} add={add} busy={busy} disabled={disabled} />
      : <CpuFallbackForm role={role} add={add} busy={busy} disabled={disabled} />;
  }
  return role.gpu
    ? <GpuCloudForm role={role} offers={offers.gpu} add={add} busy={busy} disabled={disabled} />
    : <CpuCloudForm role={role} offers={offers.cpu} add={add} busy={busy} disabled={disabled} />;
}

// The add-worker forms for one role: a local form (threads), an ssh form (an
// operator-owned machine), and/or a cloud form (a live instance selector),
// per the role's declared kinds. A singleton role's forms disable once it has
// a slot. Adding only records a paused slot; nothing launches (and no pod is
// created) until the operator starts it from the workers table.
function AddWorkerForms({ workload, role, tag, taken, onError, onChanged }: {
  workload: Workload; role: Role; tag: string; taken: boolean;
  onError: (e: string) => void; onChanged: () => void;
}) {
  // Which form is mid-request ('local' | 'cloud' | 'ssh' | null): its button
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
    <div style={{ display: 'flex', gap: 40, flexWrap: 'wrap', marginTop: 12, alignItems: 'flex-start' }}>
      <span style={{ fontSize: 13, fontWeight: 600, minWidth: 90, marginTop: 4 }} title={role.singleton ? 'at most one worker' : undefined}>
        {role.title}{role.singleton ? ' (singleton)' : ''}
      </span>
      {role.kinds.includes('local') && <LocalForm add={add} busy={busy} disabled={disabled} />}
      {role.kinds.includes('ssh') && <SshForm add={add} busy={busy} disabled={disabled} />}
      {role.kinds.includes('cloud') && <CloudForm role={role} add={add} busy={busy} disabled={disabled} />}
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
          {['worker', 'role', 'kind', 'resources', 'state', '$/hr', 'connect', ''].map((h) => (
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
          // whose pod creation keeps failing (stuck `starting`) it is the only
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
                  </div>
                )}
              </td>
              <td style={{ padding: '6px 14px 6px 0' }}>
                {w.cost_per_hr != null ? `$${w.cost_per_hr}` : '—'}
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
