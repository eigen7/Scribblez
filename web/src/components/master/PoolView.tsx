import { useCallback, useEffect, useState } from 'react';
import { getJSON, postJSON } from '../../lib/api';
import { Button } from './MasterApp';

// The machine pool (docs/plans/tag_queue.md §2): the machines the tag queue
// places tags on, owned by the pool rather than by any tag. Each row shows the
// hardware placement checks against, the tag leasing it, and any slots outside
// that lease keeping it busy (a tag placed on it by hand). Fed by GET
// /api/pool, which reads only what the reconcile pass last observed.

export type PoolMachine = {
  name: string;
  kind: 'local' | 'ssh';
  machine: { host: string; identity_file: string | null } | null;  // ssh only
  aliases: string[];
  hardware: { vcpus: number | null; gpu_count: number | null; gpu_memory_gb: number | null };
  gpu_reserve_gb: number;
  generator_threads: number | null;
  lease: { workload: string; tag: string; phase: string; since: number } | null;
  occupants: string[];  // "<workload>/<tag>/<worker_id>" of slots outside the lease
  state: 'free' | 'busy' | 'leased';
};

const POLL_MS = 5000;
const cell = { padding: '6px 14px 6px 0', fontSize: 13.5, verticalAlign: 'top' } as const;
const inputStyle = { fontSize: 13, padding: '3px 6px', border: '1px solid #b8c4d0', borderRadius: 4 };

export function gpuText(h: PoolMachine['hardware']): string {
  if (h.gpu_count == null) return '—';
  if (h.gpu_count === 0) return 'none';
  return `${h.gpu_count} × ${(h.gpu_memory_gb ?? 0).toFixed(1)} GiB`;
}

// The lease, and any slots outside it on the same machine: a tag placed there
// by hand alongside a leased one is a double booking the operator should see.
function stateText(m: PoolMachine): string {
  const others = m.occupants.join(', ');
  if (m.lease) {
    const leased = `${m.lease.workload}/${m.lease.tag} (${m.lease.phase})`;
    return others ? `${leased}; also ${others}` : leased;
  }
  return others ? `busy: ${others}` : 'free';
}

// Add this machine (host left blank) or a registered ssh machine. The server
// probes its vCPUs and GPU memory before recording it.
function AddForm({ post, busy }: { post: (url: string, body: unknown) => void; busy: boolean }) {
  const [name, setName] = useState('');
  const [host, setHost] = useState('');
  const [aliases, setAliases] = useState('');
  const [reserve, setReserve] = useState('0');
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap', marginTop: 12 }}>
      <label style={{ fontSize: 13 }}>name<br />
        <input style={{ ...inputStyle, width: 130 }} aria-label="pool name" value={name}
          onChange={(e) => setName(e.target.value)} placeholder="asus-laptop" />
      </label>
      <label style={{ fontSize: 13 }} title="blank: this machine, running local slots">ssh host<br />
        <input style={{ ...inputStyle, width: 170 }} aria-label="pool host" value={host}
          onChange={(e) => setHost(e.target.value)} placeholder="(this machine)" />
      </label>
      <label style={{ fontSize: 13 }} title="other spellings tags use for this host, comma-separated">aliases<br />
        <input style={{ ...inputStyle, width: 170 }} aria-label="pool aliases" value={aliases}
          onChange={(e) => setAliases(e.target.value)} placeholder="dshin@asus-laptop" />
      </label>
      <label style={{ fontSize: 13 }} title="GPU memory (GiB) taken by things no slot accounts for">GPU reserve<br />
        <input style={{ ...inputStyle, width: 60 }} aria-label="pool reserve" value={reserve}
          onChange={(e) => setReserve(e.target.value)} />
      </label>
      <Button
        label={busy ? 'Probing…' : 'Add to pool'}
        disabled={busy || !name.trim()}
        onClick={() => post('/api/pool/machines', {
          name: name.trim(), host: host.trim() || null, aliases, gpu_reserve_gb: reserve,
        })}
      />
    </div>
  );
}

// The operator-set fields of one machine, edited in place of its row's cells.
function EditRow({ m, post, busy, onDone }: {
  m: PoolMachine; post: (url: string, body: unknown) => void; busy: boolean; onDone: () => void;
}) {
  const [aliases, setAliases] = useState(m.aliases.join(', '));
  const [reserve, setReserve] = useState(String(m.gpu_reserve_gb));
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
        <input style={{ ...inputStyle, width: 55 }} aria-label={`${m.name} reserve`} value={reserve}
          onChange={(e) => setReserve(e.target.value)} />
      </td>
      <td style={cell}>
        <input style={{ ...inputStyle, width: 55 }} aria-label={`${m.name} threads`} value={threads}
          placeholder="auto" onChange={(e) => setThreads(e.target.value)} />
      </td>
      <td style={cell}>{stateText(m)}</td>
      <td style={{ ...cell, whiteSpace: 'nowrap' }}>
        <span style={{ display: 'inline-flex', gap: 6 }}>
          <Button label="Save" tone="primary" disabled={busy} onClick={() => {
            post('/api/pool/machine_action', {
              name: m.name, action: 'edit', aliases, gpu_reserve_gb: reserve, generator_threads: threads,
            });
            onDone();
          }} />
          <Button label="Cancel" disabled={busy} onClick={onDone} />
        </span>
      </td>
    </tr>
  );
}

export default function PoolView() {
  const [machines, setMachines] = useState<PoolMachine[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);

  const refresh = useCallback(() => {
    getJSON('/api/pool').then((d) => setMachines(d.machines)).catch((e) => setError(String(e.message ?? e)));
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
              {['machine', 'host', 'vCPUs', 'GPU', 'reserve', 'gen. threads', 'state', ''].map((h) => (
                <th key={h} style={{ ...cell, textAlign: 'left', color: '#556070', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {machines.map((m) => editing === m.name ? (
              <EditRow key={m.name} m={m} post={post} busy={busy} onDone={() => setEditing(null)} />
            ) : (
              <tr key={m.name} style={{ borderTop: '1px solid #e6eaef' }}>
                <td style={cell}><b>{m.name}</b></td>
                <td style={cell}>
                  {m.machine ? m.machine.host : 'this machine'}
                  {m.aliases.length > 0 && <div style={{ color: '#7c8694' }}>also {m.aliases.join(', ')}</div>}
                </td>
                <td style={cell}>{m.hardware.vcpus ?? '—'}</td>
                <td style={cell}>{gpuText(m.hardware)}</td>
                <td style={cell}>{m.gpu_reserve_gb ? `${m.gpu_reserve_gb} GiB` : '—'}</td>
                <td style={cell}>{m.generator_threads ?? 'auto'}</td>
                <td style={cell} data-testid={`pool-state-${m.name}`}>{stateText(m)}</td>
                <td style={{ ...cell, whiteSpace: 'nowrap' }}>
                  <span style={{ display: 'inline-flex', gap: 6 }}>
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
      <AddForm post={post} busy={busy} />
    </div>
  );
}
