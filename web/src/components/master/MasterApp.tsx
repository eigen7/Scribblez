import { useCallback, useEffect, useState } from 'react';
import { getJSON, postJSON } from '../../lib/api';
import TaskView from './TaskView';

// The master dashboard: the entrypoint for all work. Pick a workload, then a
// tag (or create one, configuring its parameters via the workload's schema);
// a tag opens its task view (workers, progress, workload tabs). Rendered for
// VITE_TOOL=dashboard. See docs/master_dashboard.md.

export type ParamField = {
  name: string; kind: 'int' | 'float' | 'str' | 'bool';
  default: number | boolean | string; help: string;
  choices: string[] | null;  // the closed set of accepted values; null if open
};
export type Role = {
  name: string; title: string; singleton: boolean; kinds: string[]; interruptible: boolean;
  gpu: boolean;
  stats: { unit: string; phases: Record<string, string> } | null;
};
export type Workload = {
  name: string; title: string; params: ParamField[]; roles: Role[];
  primary_params: string[];  // shown up front by the new-tag form; the rest are advanced
};
type TagRow = {
  tag: string; has_task: boolean; created_at: number | null;
  workers: number; progress: [string, string | number][]; last_active: number;
};

export function relTime(epochSeconds: number | null): string {
  if (!epochSeconds) return '—';
  const s = Date.now() / 1000 - epochSeconds;
  if (s < 90) return `${Math.max(0, Math.round(s))}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 129600) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

const inputStyle = { fontSize: 14, padding: '3px 6px', border: '1px solid #b8c4d0', borderRadius: 4 };

export function Button({ label, onClick, disabled, tone }: {
  label: string; onClick: () => void; disabled?: boolean; tone?: 'danger' | 'primary';
}) {
  const bg = disabled ? '#b8c4d0' : tone === 'danger' ? '#b23b3b' : '#1f77b4';
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        fontSize: 13, padding: '4px 12px', border: 'none', borderRadius: 4,
        background: bg, color: 'white', cursor: disabled ? 'default' : 'pointer',
      }}
    >
      {label}
    </button>
  );
}

// One control per schema field: a checkbox for a bool, a selector for a field
// whose value set is closed, a text box otherwise.
function ParamInput({ p, value, bad, onChange }: {
  p: ParamField; value: string | boolean; bad: boolean; onChange: (v: string | boolean) => void;
}) {
  if (p.kind === 'bool') {
    return (
      <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
    );
  }
  const style = {
    ...inputStyle, width: p.kind === 'str' ? 160 : 90,
    borderColor: bad ? '#b23b3b' : '#b8c4d0',
  };
  if (p.choices) {
    return (
      <select style={style} value={String(value ?? '')} onChange={(e) => onChange(e.target.value)}>
        {p.choices.map((c) => <option key={c} value={c}>{c}</option>)}
      </select>
    );
  }
  return (
    <input style={style} value={String(value ?? '')} onChange={(e) => onChange(e.target.value)} />
  );
}

const paramRowStyle = {
  display: 'flex', gap: 22, flexWrap: 'wrap', alignItems: 'flex-end',
} as const;

// One labelled control per field, hovering its help text.
function ParamFields({ params, values, bad, onChange }: {
  params: ParamField[];
  values: Record<string, string | boolean>;
  bad: ParamField[];
  onChange: (name: string, v: string | boolean) => void;
}) {
  return (
    <>
      {params.map((p) => (
        <label key={p.name} style={{ fontSize: 13 }} title={p.help}>
          {p.name}<br />
          <ParamInput
            p={p}
            value={values[p.name] ?? ''}
            bad={bad.some((b) => b.name === p.name)}
            onChange={(v) => onChange(p.name, v)}
          />
        </label>
      ))}
    </>
  );
}

// The fields the workload names as its up-front ones, in the order it names
// them, and everything else. A workload naming none shows all of them up front.
function splitParams(workload: Workload): [ParamField[], ParamField[]] {
  const byName = new Map(workload.params.map((p) => [p.name, p]));
  const primary = workload.primary_params.flatMap((n) => {
    const p = byName.get(n);
    return p ? [p] : [];
  });
  if (primary.length === 0) return [workload.params, []];
  const upFront = new Set(primary.map((p) => p.name));
  return [primary, workload.params.filter((p) => !upFront.has(p.name))];
}

// The new-tag section: a tag-name input plus one control per schema field,
// validated client-side (numbers must parse) and server-side (the create
// endpoint re-validates and reports errors verbatim).
//
// Only the workload's up-front fields — the handful that decide what a run
// is — are shown; the long tail sits behind the "Advanced" disclosure at its
// schema defaults. A field left invalid inside the collapsed section forces it
// open, so Create is never disabled by something the operator cannot see.
export function NewTagForm({ workload, onCreated }: { workload: Workload; onCreated: (tag: string) => void }) {
  const [tag, setTag] = useState('');
  const [values, setValues] = useState<Record<string, string | boolean>>({});
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  useEffect(() => {
    setValues(Object.fromEntries(workload.params.map(
      (p) => [p.name, p.kind === 'bool' ? Boolean(p.default) : String(p.default)],
    )));
  }, [workload]);

  const [primary, advanced] = splitParams(workload);
  const numberOk = (p: ParamField, raw: string) =>
    p.kind === 'int' ? /^-?\d+$/.test(raw) : !Number.isNaN(parseFloat(raw));
  const tagOk = /^[A-Za-z0-9._-]+$/.test(tag);
  const badNumbers = workload.params.filter(
    (p) => (p.kind === 'int' || p.kind === 'float') && !numberOk(p, String(values[p.name] ?? '')),
  );
  const canCreate = tagOk && badNumbers.length === 0 && !busy;
  const showAdvanced = advancedOpen || advanced.some(
    (p) => badNumbers.some((b) => b.name === p.name),
  );

  const setValue = (name: string, v: string | boolean) =>
    setValues((s) => ({ ...s, [name]: v }));

  const coerce = (p: ParamField) => {
    const raw = values[p.name];
    if (p.kind === 'bool') return Boolean(raw);
    if (p.kind === 'str') return String(raw ?? '');
    if (p.kind === 'float') return parseFloat(String(raw));
    return parseInt(String(raw), 10);
  };

  const create = async () => {
    setBusy(true);
    setError('');
    const params = Object.fromEntries(workload.params.map((p) => [p.name, coerce(p)]));
    try {
      await postJSON('/api/tasks', { workload: workload.name, tag, params });
      onCreated(tag);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card" style={{ marginTop: 14 }}>
      <b style={{ fontSize: 15 }}>New tag</b>
      <div style={{ ...paramRowStyle, marginTop: 10 }}>
        <label style={{ fontSize: 13 }}>
          Tag name<br />
          <input
            style={{ ...inputStyle, borderColor: tag && !tagOk ? '#b23b3b' : '#b8c4d0' }}
            value={tag}
            onChange={(e) => setTag(e.target.value.trim())}
            placeholder="e.g. exp42"
          />
        </label>
        <ParamFields params={primary} values={values} bad={badNumbers} onChange={setValue} />
        <Button label={busy ? 'Creating…' : 'Create'} onClick={create} disabled={!canCreate} />
      </div>
      {advanced.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <span
            onClick={() => setAdvancedOpen(!showAdvanced)}
            style={{ fontSize: 13, color: '#1f77b4', cursor: 'pointer', userSelect: 'none' }}
          >
            {showAdvanced ? '▾' : '▸'} Advanced: {advanced.length} more settings
          </span>
          {showAdvanced && (
            <div style={{ ...paramRowStyle, marginTop: 10 }}>
              <ParamFields params={advanced} values={values} bad={badNumbers} onChange={setValue} />
            </div>
          )}
        </div>
      )}
      <div style={{ fontSize: 12, color: '#556070', marginTop: 8 }}>
        Parameters are frozen at creation: every worker on a tag generates with identical settings.
        Hover a field for its meaning.
      </div>
      {error && <div style={{ color: '#b23b3b', fontSize: 13, marginTop: 8 }}>{error}</div>}
    </div>
  );
}

function HomePage({ workload, onOpen }: { workload: Workload; onOpen: (tag: string) => void }) {
  const [rows, setRows] = useState<TagRow[]>([]);
  const [sortBy, setSortBy] = useState<'name' | 'last-ran'>('last-ran');
  const [error, setError] = useState('');

  const refresh = useCallback(() => {
    getJSON(`/api/workload_tags?workload=${workload.name}`)
      .then((d) => setRows(d.tags))
      .catch(() => {});
  }, [workload]);
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);

  // Deleting a tag deletes its local data dir (the bucket archive, if any, is
  // kept); the server refuses while the tag still has workers.
  const deleteTag = async (tag: string) => {
    if (!window.confirm(`Delete tag '${tag}' and its local data?`)) return;
    setError('');
    try {
      await postJSON('/api/task/delete', { workload: workload.name, tag });
      refresh();
    } catch (e) {
      setError(String(e));
    }
  };

  const sorted = [...rows].sort((a, b) =>
    sortBy === 'name' ? a.tag.localeCompare(b.tag) : b.last_active - a.last_active,
  );

  return (
    <>
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 10 }}>
          <b style={{ fontSize: 15 }}>Tags</b>
          <label style={{ fontSize: 13 }}>
            sort by{' '}
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value as 'name' | 'last-ran')}>
              <option value="last-ran">last ran</option>
              <option value="name">name</option>
            </select>
          </label>
        </div>
        {sorted.length === 0 ? (
          <div style={{ color: '#556070', fontStyle: 'italic' }}>No tags yet — create one below.</div>
        ) : (
          <table style={{ borderCollapse: 'collapse', fontSize: 14, width: '100%' }}>
            <thead>
              <tr style={{ textAlign: 'left', color: '#445063' }}>
                <th style={{ padding: '4px 14px 4px 0' }}>tag</th>
                <th style={{ padding: '4px 14px 4px 0' }}>progress</th>
                <th style={{ padding: '4px 14px 4px 0' }}>workers</th>
                <th style={{ padding: '4px 14px 4px 0' }}>last ran</th>
                <th style={{ padding: '4px 14px 4px 0' }} />
                <th style={{ padding: '4px 14px 4px 0' }} />
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr
                  key={r.tag}
                  onClick={() => onOpen(r.tag)}
                  style={{ cursor: 'pointer', borderTop: '1px solid #e2e8ee' }}
                >
                  <td style={{ padding: '6px 14px 6px 0', fontWeight: 600 }}>{r.tag}</td>
                  <td style={{ padding: '6px 14px 6px 0' }}>
                    {r.progress.map(([k, v]) => `${k}: ${v}`).join(' · ') || '—'}
                  </td>
                  <td style={{ padding: '6px 14px 6px 0' }}>{r.workers}</td>
                  <td style={{ padding: '6px 14px 6px 0' }}>{relTime(r.last_active)}</td>
                  <td style={{ padding: '6px 14px 6px 0', fontSize: 12, color: '#8494a5' }}>
                    {r.has_task ? '' : 'pre-dashboard tag (read-only)'}
                  </td>
                  <td style={{ padding: '6px 0' }} onClick={(e) => e.stopPropagation()}>
                    <span title={r.workers > 0 ? 'remove the tag’s workers first' : undefined}>
                      <Button
                        label="Delete" tone="danger" disabled={r.workers > 0}
                        onClick={() => deleteTag(r.tag)}
                      />
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {error && <div style={{ color: '#b23b3b', fontSize: 13, marginTop: 8 }}>{error}</div>}
      </div>
      <NewTagForm workload={workload} onCreated={onOpen} />
    </>
  );
}

export default function MasterApp() {
  const initial = new URLSearchParams(window.location.search);
  const [workloads, setWorkloads] = useState<Workload[]>([]);
  const [workloadName, setWorkloadName] = useState(initial.get('workload') ?? 'kill_test');
  const [openTag, setOpenTag] = useState<string | null>(initial.get('tag'));

  useEffect(() => {
    getJSON('/api/workloads').then((d) => setWorkloads(d.workloads)).catch(() => {});
  }, []);

  // Keep ?workload=&tag= in the URL so views are shareable and reload-stable.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set('workload', workloadName);
    if (openTag) params.set('tag', openTag);
    else params.delete('tag');
    window.history.replaceState(null, '', `${window.location.pathname}?${params}`);
  }, [workloadName, openTag]);

  const workload = workloads.find((w) => w.name === workloadName);

  return (
    <div style={{
      fontFamily: 'system-ui, sans-serif', padding: '14px 18px', color: '#1a1f28', fontSize: 15,
      background: '#f4f6f8', minHeight: '100vh',
    }}>
      <div className="page-cap">
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 14 }}>
          <strong style={{ fontSize: 19 }}>Scribblez</strong>
          <label style={{ fontSize: 15 }}>
            Workload{' '}
            <select
              value={workloadName}
              onChange={(e) => { setWorkloadName(e.target.value); setOpenTag(null); }}
            >
              {workloads.map((w) => (
                <option key={w.name} value={w.name}>{w.title}</option>
              ))}
            </select>
          </label>
          {openTag && (
            <>
              <span
                onClick={() => setOpenTag(null)}
                style={{ color: '#1f77b4', cursor: 'pointer', fontSize: 14 }}
              >
                ← all tags
              </span>
              <b>{openTag}</b>
            </>
          )}
        </div>
        {!workload ? (
          <div style={{ color: '#556070', fontStyle: 'italic' }}>Connecting to the data API…</div>
        ) : openTag ? (
          <TaskView workload={workload} tag={openTag} />
        ) : (
          <HomePage workload={workload} onOpen={setOpenTag} />
        )}
      </div>
    </div>
  );
}
