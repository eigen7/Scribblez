import { useCallback, useEffect, useState } from 'react';
import { getJSON, postJSON } from '../../lib/api';
import { Button } from './MasterApp';
import { RentalOffer, rateText } from './rentalOffer';
import { TagLink } from './ui';

// The machine pool (docs/plans/tag_queue.md §2): the machines the tag queue
// places tags on, owned by the pool rather than by any tag. Each row shows the
// hardware placement checks against, the tag leasing it, and any slots outside
// that lease keeping it busy (a tag placed on it by hand). Fed by GET
// /api/pool, which reads only what the reconcile pass last observed.

export type PoolMachine = {
  name: string;
  kind: 'local' | 'ssh';
  // ssh only; a rented one also carries its instance and spend.
  machine: {
    host: string; identity_file: string | null; instance_type?: string | null;
    spot?: boolean; spend?: number;
  } | null;
  aliases: string[];
  hardware: { vcpus: number | null; gpu_count: number | null; gpu_memory_gb: number | null };
  generator_threads: number | null;
  lease: { workload: string; tag: string; phase: string; since: number; reason: string } | null;
  occupants: string[];  // "<workload>/<tag>/<worker_id>" of slots outside the lease
  state: 'free' | 'busy' | 'leased';
  capacity: string | null;  // the capacity entry a machine the pool rented belongs to
  retiring: boolean;  // being given up (Stop all cloud spending)
};

// Machines the pool may rent (pool.Capacity): up to `cap` of `instance_type`.
export type Capacity = { name: string; instance_type: string; spot: boolean; cap: number };

const POLL_MS = 5000;
const cell = { padding: '6px 14px 6px 0', fontSize: 13.5, verticalAlign: 'top' } as const;
const inputStyle = { fontSize: 13, padding: '3px 6px', border: '1px solid #b8c4d0', borderRadius: 4 };

export function gpuText(h: PoolMachine['hardware']): string {
  if (h.gpu_count == null) return '—';
  if (h.gpu_count === 0) return 'none';
  return `${h.gpu_count} × ${(h.gpu_memory_gb ?? 0).toFixed(1)} GiB`;
}

type OpenTag = (workload: string, tag: string) => void;

// The lease, and any slots outside it on the same machine: a tag placed there
// by hand alongside a leased one is a double booking the operator should see.
// Each tag named opens its task view.
function StateText({ m, onOpenTag }: { m: PoolMachine; onOpenTag: OpenTag }) {
  return (
    <>
      <BaseState m={m} onOpenTag={onOpenTag} />
      {m.retiring && ' — retiring: terminated once free'}
    </>
  );
}

function BaseState({ m, onOpenTag }: { m: PoolMachine; onOpenTag: OpenTag }) {
  const others = <Occupants occupants={m.occupants} onOpenTag={onOpenTag} />;
  if (m.lease) {
    const why = m.lease.reason ? `: ${m.lease.reason}` : '';
    return (
      <>
        <TagLink workload={m.lease.workload} tag={m.lease.tag} onOpen={onOpenTag} /> ({m.lease.phase}{why})
        {m.occupants.length > 0 && <>; also {others}</>}
      </>
    );
  }
  return m.occupants.length > 0 ? <>busy: {others}</> : <>free</>;
}

// Slots as "<workload>/<tag>/<worker_id>", the workload/tag part linked.
function Occupants({ occupants, onOpenTag }: { occupants: string[]; onOpenTag: OpenTag }) {
  return occupants.map((o, i) => {
    const [workload, tag, worker] = o.split('/');
    return (
      <span key={o}>
        {i > 0 && ', '}
        <TagLink workload={workload} tag={tag} onOpen={onOpenTag} />/{worker}
      </span>
    );
  });
}

// One queued tag (dashboard/tag_queue.py's status): its order, whether it
// ends on its own, its bundle, and why each pool machine did not take it.
export type QueueRow = {
  workload: string; tag: string; machines: string[]; memory_override_gb: number | null;
  bundle: string; end_condition: boolean; refusals: Record<string, string>;
};

// Queue a tag, asking the operator to confirm when the server has warnings (a
// tag without an end condition, local slots on the live checkout). Resolves to
// whether it was queued.
export async function enqueueTag(workload: string, tag: string): Promise<boolean> {
  const first = await postJSON('/api/queue/enqueue', { workload, tag });
  if (first.queued) return true;
  const ok = window.confirm(`Queue ${workload}/${tag}?\n\n${first.warnings.join('\n\n')}`);
  if (!ok) return false;
  await postJSON('/api/queue/enqueue', { workload, tag, confirm: true });
  return true;
}

function QueueSection({ rows, post, busy, onOpenTag }: {
  rows: QueueRow[]; post: (url: string, body: unknown) => void; busy: boolean; onOpenTag: OpenTag;
}) {
  const act = (r: QueueRow, action: string) =>
    post('/api/queue/action', { workload: r.workload, tag: r.tag, action });
  return (
    <div style={{ marginTop: 18 }}>
      <strong>Queue</strong>
      <div style={{ fontSize: 13, color: '#556070', margin: '4px 0 10px' }}>
        Placed in order on the first free machine each fits; ∞ marks a tag with no end
        condition, which holds its machine until released.
      </div>
      {rows.length === 0 ? (
        <div style={{ fontStyle: 'italic', color: '#556070' }}>Nothing queued.</div>
      ) : (
        <table style={{ borderCollapse: 'collapse' }}>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.workload}/${r.tag}`} style={{ borderTop: '1px solid #e6eaef' }}>
                <td style={cell}>{i + 1}</td>
                <td style={cell} data-testid={`queue-${r.tag}`}>
                  <b><TagLink workload={r.workload} tag={r.tag} onOpen={onOpenTag} /></b>{r.end_condition ? '' : ' ∞'}
                  {r.machines.length > 0 && <div style={{ color: '#7c8694' }}>only {r.machines.join(', ')}</div>}
                </td>
                <td style={cell}>bundle {r.bundle}</td>
                <td style={{ ...cell, color: '#556070', maxWidth: 520 }}>
                  {Object.entries(r.refusals).map(([m, why]) => (
                    <div key={m}>{m}: {why}</div>
                  ))}
                </td>
                <td style={{ ...cell, whiteSpace: 'nowrap' }}>
                  <span style={{ display: 'inline-flex', gap: 6 }}>
                    <Button label="↑" disabled={busy || i === 0} onClick={() => act(r, 'up')} />
                    <Button label="↓" disabled={busy || i === rows.length - 1} onClick={() => act(r, 'down')} />
                    <Button label="Dequeue" tone="danger" disabled={busy} onClick={() => act(r, 'dequeue')} />
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// The capacity entries, each with its rented machines counted, a cap editor,
// and a form to add one from the provider's catalog.
function CapacitySection({ capacity, machines, post, busy }: {
  capacity: Capacity[]; machines: PoolMachine[];
  post: (url: string, body: unknown) => void; busy: boolean;
}) {
  const [offer, setOffer] = useState<RentalOffer | null>(null);
  // Why the catalog could not be read (no cloud credentials, say), as the task
  // view's rent form reports it.
  const [unavailable, setUnavailable] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [typeId, setTypeId] = useState('');
  const [spot, setSpot] = useState(true);
  const [cap, setCap] = useState('1');
  useEffect(() => {
    getJSON('/api/cloud/rental_offer')
      .then((d: RentalOffer) => {
        const gpus = d.types.filter((t) => t.gpu_count > 0);
        setOffer({ ...d, types: gpus });
        if (gpus.length) setTypeId(gpus[0].id);
        else setUnavailable('the provider offers no GPU types');
      })
      .catch((e) => setUnavailable(String(e.message ?? e)));
  }, []);
  const act = (body: object) => post('/api/pool/capacity', body);
  return (
    <div style={{ marginTop: 18 }}>
      <strong>Rental capacity</strong>
      <div style={{ fontSize: 13, color: '#556070', margin: '4px 0 10px' }}>
        The queue rents here on its own, whenever a queued tag fits no free machine above, up to
        each entry's cap; a rented machine is terminated once idle. Removing a rented machine
        does not stop this: set the cap to 0, or remove the entry. Set the cap to what your
        quota allows.
      </div>
      {capacity.length > 0 && (
        <table style={{ borderCollapse: 'collapse', marginBottom: 8 }}>
          <tbody>
            {capacity.map((c) => {
              const rented = machines.filter((m) => m.capacity === c.name).length;
              return (
                <tr key={c.name} style={{ borderTop: '1px solid #e6eaef' }}>
                  <td style={cell}><b>{c.name}</b></td>
                  <td style={cell}>{c.instance_type}{c.spot ? ' spot' : ''}{offer && `, ${rateText(offer, c.instance_type, c.spot)}`}</td>
                  <td style={cell} data-testid={`capacity-${c.name}`}>{rented} of {c.cap} rented</td>
                  <td style={{ ...cell, whiteSpace: 'nowrap' }}>
                    <span style={{ display: 'inline-flex', gap: 6 }}>
                      <Button label="−" disabled={busy || c.cap === 0}
                        onClick={() => act({ action: 'set_cap', name: c.name, cap: c.cap - 1 })} />
                      <Button label="+" disabled={busy}
                        onClick={() => act({ action: 'set_cap', name: c.name, cap: c.cap + 1 })} />
                      <Button label="Remove" tone="danger" disabled={busy}
                        onClick={() => act({ action: 'remove', name: c.name })} />
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {unavailable && (
        <div style={{ fontSize: 13, color: '#a05a00', marginBottom: 6 }}>
          renting unavailable: {unavailable}
        </div>
      )}
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <label style={{ fontSize: 13 }}>name<br />
          <input style={{ ...inputStyle, width: 90 }} aria-label="capacity name" value={name}
            onChange={(e) => setName(e.target.value)} placeholder="g6" />
        </label>
        <label style={{ fontSize: 13 }}>type<br />
          <select style={inputStyle} aria-label="capacity type" value={typeId} onChange={(e) => setTypeId(e.target.value)}>
            {offer?.types.map((t) => (
              <option key={t.id} value={t.id}>{t.id} ({t.vcpus} vCPU, {t.gpu}, {rateText(offer, t.id, spot)})</option>
            ))}
          </select>
        </label>
        <label style={{ fontSize: 13 }}>
          <input type="checkbox" aria-label="capacity spot" checked={spot} onChange={(e) => setSpot(e.target.checked)} /> spot
        </label>
        <label style={{ fontSize: 13 }}>cap<br />
          <input style={{ ...inputStyle, width: 40 }} aria-label="capacity cap" value={cap}
            onChange={(e) => setCap(e.target.value)} />
        </label>
        <Button label="Add capacity" disabled={busy || !name.trim() || !typeId || !cap.trim()}
          onClick={() => act({ action: 'add', name: name.trim(), instance_type: typeId, spot, cap })} />
      </div>
    </div>
  );
}

// Add this machine in one click, while it is not pooled, or a registered ssh
// machine. The server probes its vCPUs and GPU memory before recording it.
function AddForm({ post, busy, localPooled }: {
  post: (url: string, body: unknown) => void; busy: boolean; localPooled: boolean;
}) {
  const [name, setName] = useState('');
  const [host, setHost] = useState('');
  const [aliases, setAliases] = useState('');
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap', marginTop: 12 }}>
      {!localPooled && (
        <Button label="Add this machine" disabled={busy}
          onClick={() => post('/api/pool/machines', { name: 'localhost', host: null })} />
      )}
      <label style={{ fontSize: 13 }}>name<br />
        <input style={{ ...inputStyle, width: 130 }} aria-label="pool name" value={name}
          onChange={(e) => setName(e.target.value)} placeholder="asus-laptop" />
      </label>
      <label style={{ fontSize: 13 }}>ssh host<br />
        <input style={{ ...inputStyle, width: 170 }} aria-label="pool host" value={host}
          onChange={(e) => setHost(e.target.value)} placeholder="asus-laptop" />
      </label>
      <label style={{ fontSize: 13 }} title="other spellings tags use for this host, comma-separated">aliases<br />
        <input style={{ ...inputStyle, width: 170 }} aria-label="pool aliases" value={aliases}
          onChange={(e) => setAliases(e.target.value)} placeholder="dshin@asus-laptop" />
      </label>
      <Button
        label={busy ? 'Probing…' : 'Add ssh machine'}
        disabled={busy || !name.trim() || !host.trim()}
        onClick={() => post('/api/pool/machines', { name: name.trim(), host: host.trim(), aliases })}
      />
    </div>
  );
}

// The operator-set fields of one machine, edited in place of its row's cells.
function EditRow({ m, post, busy, onDone, onOpenTag }: {
  m: PoolMachine; post: (url: string, body: unknown) => void; busy: boolean; onDone: () => void;
  onOpenTag: OpenTag;
}) {
  const [aliases, setAliases] = useState(m.aliases.join(', '));
  const [threads, setThreads] = useState(m.generator_threads == null ? '' : String(m.generator_threads));
  return (
    <tr style={{ borderTop: '1px solid #e6eaef', background: '#f7f9fb' }}>
      <td style={cell}><b>{m.name}</b></td>
      <td style={cell}>
        <input style={{ ...inputStyle, width: 170 }} aria-label={`${m.name} aliases`} value={aliases}
          onChange={(e) => setAliases(e.target.value)} />
      </td>
      <td style={cell}>{m.hardware.vcpus ?? '—'}</td>
      <td style={cell}>{gpuText(m.hardware)}</td>
      <td style={cell}>
        <input style={{ ...inputStyle, width: 55 }} aria-label={`${m.name} threads`} value={threads}
          placeholder="auto" onChange={(e) => setThreads(e.target.value)} />
      </td>
      <td style={cell}><StateText m={m} onOpenTag={onOpenTag} /></td>
      <td style={{ ...cell, whiteSpace: 'nowrap' }}>
        <span style={{ display: 'inline-flex', gap: 6 }}>
          <Button label="Save" tone="primary" disabled={busy} onClick={() => {
            post('/api/pool/machine_action', {
              name: m.name, action: 'edit', aliases, generator_threads: threads,
            });
            onDone();
          }} />
          <Button label="Cancel" disabled={busy} onClick={onDone} />
        </span>
      </td>
    </tr>
  );
}

export default function PoolView({ onOpenTag }: { onOpenTag: OpenTag }) {
  const [machines, setMachines] = useState<PoolMachine[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [queue, setQueue] = useState<QueueRow[]>([]);
  const [capacity, setCapacity] = useState<Capacity[]>([]);

  const refresh = useCallback(() => {
    getJSON('/api/pool')
      .then((d) => { setMachines(d.machines); setCapacity(d.capacity ?? []); })
      .catch((e) => setError(String(e.message ?? e)));
    getJSON('/api/queue').then((d) => setQueue(d.entries)).catch((e) => setError(String(e.message ?? e)));
  }, []);
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  const post = (url: string, body: unknown) => {
    setBusy(true);
    setError(null);
    postJSON(url, body)
      .then(refresh)
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setBusy(false));
  };

  return (
    <div style={{ background: 'white', borderRadius: 6, padding: '12px 16px', border: '1px solid #d3d8e0' }}>
      <strong>Machine pool</strong>
      <div style={{ fontSize: 13, color: '#556070', margin: '4px 0 10px' }}>
        Machines the tag queue may place tags on, one tag at a time each.
      </div>
      {error && <div style={{ color: '#b23b3b', fontSize: 13, marginBottom: 8 }}>{error}</div>}
      {machines == null ? (
        <div style={{ fontStyle: 'italic', color: '#556070' }}>Loading…</div>
      ) : machines.length === 0 ? (
        <div style={{ fontStyle: 'italic', color: '#556070' }}>The pool is empty.</div>
      ) : (
        <table style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {['machine', 'host', 'vCPUs', 'GPU', 'gen. threads', 'state', ''].map((h) => (
                <th key={h} style={{ ...cell, textAlign: 'left', color: '#556070', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {machines.map((m) => editing === m.name ? (
              <EditRow key={m.name} m={m} post={post} busy={busy} onDone={() => setEditing(null)} onOpenTag={onOpenTag} />
            ) : (
              <tr key={m.name} style={{ borderTop: '1px solid #e6eaef' }}>
                <td style={cell}><b>{m.name}</b></td>
                <td style={cell}>
                  {m.machine ? m.machine.host : 'this machine'}
                  {m.capacity && m.machine && (
                    <div style={{ color: '#7c8694' }}>
                      rented {m.machine.instance_type}{m.machine.spot ? ' spot' : ''} by capacity {m.capacity},
                      ${(m.machine.spend ?? 0).toFixed(2)} so far
                    </div>
                  )}
                  {m.aliases.length > 0 && <div style={{ color: '#7c8694' }}>also {m.aliases.join(', ')}</div>}
                </td>
                <td style={cell}>{m.hardware.vcpus ?? '—'}</td>
                <td style={cell}>{gpuText(m.hardware)}</td>
                <td style={cell}>{m.generator_threads ?? 'auto'}</td>
                <td style={cell} data-testid={`pool-state-${m.name}`}><StateText m={m} onOpenTag={onOpenTag} /></td>
                <td style={{ ...cell, whiteSpace: 'nowrap' }}>
                  <span style={{ display: 'inline-flex', gap: 6 }}>
                    {m.lease && (
                      <>
                        <Button label="Requeue" disabled={busy || m.lease.phase === 'releasing'}
                          onClick={() => post('/api/queue/action', { workload: m.lease!.workload, tag: m.lease!.tag, action: 'requeue' })} />
                        <Button label="Release" disabled={busy || m.lease.phase !== 'running'}
                          onClick={() => {
                            if (window.confirm(`Finish every slot of ${m.lease!.tag} and release ${m.name}?`)) {
                              post('/api/queue/action', { workload: m.lease!.workload, tag: m.lease!.tag, action: 'release' });
                            }
                          }} />
                      </>
                    )}
                    <Button label="Edit" disabled={busy} onClick={() => setEditing(m.name)} />
                    <Button label="Re-probe" disabled={busy}
                      onClick={() => post('/api/pool/machine_action', { name: m.name, action: 'reprobe' })} />
                    <Button label="Remove" tone="danger" disabled={busy || m.lease != null}
                      onClick={() => {
                        if (window.confirm(`Remove ${m.name} from the pool?`)) {
                          post('/api/pool/machine_action', { name: m.name, action: 'remove' });
                        }
                      }} />
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <AddForm post={post} busy={busy} localPooled={(machines ?? []).some((m) => m.kind === 'local')} />
      <CapacitySection capacity={capacity} machines={machines ?? []} post={post} busy={busy} />
      <QueueSection rows={queue} post={post} busy={busy} onOpenTag={onOpenTag} />
    </div>
  );
}
