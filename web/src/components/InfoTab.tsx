import { useCallback, useEffect, useState } from 'react';
import { getJSON } from '../lib/api';

// The Info tab: a run's recorded configuration (the training args), model size,
// and timestamps, read from the per-tag meta row via /api/meta. The meta is
// rewritten each time a run (re)launches, so this polls to pick up a relaunch.

type Meta = {
  tag: string | null;
  model_params: number | null;
  created_at: number | null;
  updated_at: number | null;
  args: Record<string, unknown>;
};

const EMPTY: React.CSSProperties = { color: '#556070', fontStyle: 'italic', padding: 20 };

function fmtTime(epoch: number | null): string {
  return epoch ? new Date(epoch * 1000).toLocaleString() : '—';
}

function fmtValue(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (Array.isArray(v)) return v.length ? v.join(', ') : '[]';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  return String(v);
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ fontSize: 12, color: '#6b7785', textTransform: 'uppercase', letterSpacing: 0.4 }}>
        {label}
      </div>
      <div style={{ fontSize: 15, color: '#1a1f28' }}>{value}</div>
    </div>
  );
}

export default function InfoTab({ task, tag }: { task: string; tag: string | null }) {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [loaded, setLoaded] = useState(false);

  const refetch = useCallback(async () => {
    if (!tag) {
      setMeta(null);
      setLoaded(true);
      return;
    }
    try {
      const d = await getJSON(`/api/meta?task=${task}&tag=${encodeURIComponent(tag)}`);
      setMeta(d.meta ?? null);
    } catch {
      setMeta(null);
    }
    setLoaded(true);
  }, [task, tag]);

  useEffect(() => {
    setLoaded(false);
    refetch().catch(() => {});
  }, [refetch]);

  useEffect(() => {
    const id = setInterval(() => refetch().catch(() => {}), 5000);
    return () => clearInterval(id);
  }, [refetch]);

  if (!tag) return <div className="card"><div style={EMPTY}>Select a run.</div></div>;
  if (!loaded) return <div className="card"><div style={EMPTY}>Loading…</div></div>;
  if (!meta) return <div className="card"><div style={EMPTY}>No run config recorded yet.</div></div>;

  const args = meta.args ?? {};
  const keys = Object.keys(args).sort();

  return (
    <div className="card">
      <div style={{ display: 'flex', gap: 28, flexWrap: 'wrap', marginBottom: 16 }}>
        <Stat label="Tag" value={meta.tag ?? tag} />
        <Stat
          label="Model params"
          value={meta.model_params != null ? meta.model_params.toLocaleString() : '—'}
        />
        <Stat label="Started" value={fmtTime(meta.created_at)} />
        <Stat label="Updated" value={fmtTime(meta.updated_at)} />
      </div>
      <table style={{ borderCollapse: 'collapse', fontSize: 13, width: '100%', maxWidth: 760 }}>
        <thead>
          <tr style={{ textAlign: 'left', color: '#6b7785' }}>
            <th style={{ padding: '4px 12px 6px 0', fontWeight: 600 }}>Argument</th>
            <th style={{ padding: '4px 0 6px', fontWeight: 600 }}>Value</th>
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => (
            <tr key={k} style={{ borderTop: '1px solid #e2e8f0' }}>
              <td style={{ padding: '4px 12px 4px 0', color: '#445063', fontFamily: 'monospace' }}>
                {k}
              </td>
              <td style={{ padding: '4px 0', fontFamily: 'monospace' }}>{fmtValue(args[k])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
