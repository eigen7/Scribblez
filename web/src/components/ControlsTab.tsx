import { useCallback, useEffect, useState } from 'react';
import { getJSON, postJSON } from '../lib/api';

// The Controls tab: live operator knobs read from and written to the per-tag
// dashboard.db via /api/controls. Today it exposes the base learning rate, which
// the generational trainer adopts at its next epoch and records a rows-clock
// change event for. Setting it here persists across trainer restarts.

type ControlEvent = { positions: number; name: string; value: number; t: number };
type ControlsData = { controls: Record<string, number>; events: ControlEvent[] };

const EMPTY: React.CSSProperties = { color: '#556070', fontStyle: 'italic', padding: 20 };
const LABEL: React.CSSProperties = {
  fontSize: 12,
  color: '#6b7785',
  textTransform: 'uppercase',
  letterSpacing: 0.4,
};

export default function ControlsTab({ task, tag }: { task: string; tag: string | null }) {
  const [data, setData] = useState<ControlsData | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const url = tag ? `/api/controls?task=${task}&tag=${encodeURIComponent(tag)}` : null;

  const refetch = useCallback(async () => {
    if (!url) {
      setData(null);
      setLoaded(true);
      return;
    }
    try {
      setData(await getJSON(url));
    } catch {
      setData(null);
    }
    setLoaded(true);
  }, [url]);

  // Refetch on run switch, and poll (the draft input is separate, so polling
  // never clobbers what the operator is typing).
  useEffect(() => {
    setLoaded(false);
    setDraft('');
    setErr(null);
    refetch().catch(() => {});
  }, [refetch]);
  useEffect(() => {
    const id = setInterval(() => refetch().catch(() => {}), 5000);
    return () => clearInterval(id);
  }, [refetch]);

  const baseLr = data?.controls?.base_lr;

  const submit = useCallback(async () => {
    if (!url) return;
    const value = Number(draft);
    if (!draft.trim() || !Number.isFinite(value) || value <= 0) {
      setErr('Enter a positive number (e.g. 2e-4).');
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      await postJSON(url, { name: 'base_lr', value });
      setDraft('');
      await refetch();
    } catch (e) {
      setErr(String(e));
    }
    setBusy(false);
  }, [url, draft, refetch]);

  if (!tag) {
    return (
      <div className="card">
        <div style={EMPTY}>Select a run.</div>
      </div>
    );
  }
  if (!loaded) {
    return (
      <div className="card">
        <div style={EMPTY}>Loading…</div>
      </div>
    );
  }

  const events = (data?.events ?? []).filter((e) => e.name === 'base_lr');

  return (
    <div className="card">
      <div style={{ marginBottom: 16 }}>
        <div style={LABEL}>Base learning rate</div>
        <div style={{ fontSize: 22, color: '#1a1f28', fontFamily: 'monospace' }}>
          {baseLr != null ? baseLr.toExponential(3) : '—'}
        </div>
        <div style={{ fontSize: 12, color: '#6b7785', marginTop: 4 }}>
          The trainer adopts a new value at its next epoch; it persists across restarts.
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
        <input
          type="text"
          value={draft}
          placeholder={baseLr != null ? baseLr.toExponential(3) : 'e.g. 2e-4'}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') submit();
          }}
          style={{ fontFamily: 'monospace', width: 160, padding: '4px 8px' }}
        />
        <button onClick={submit} disabled={busy}>
          Set
        </button>
      </div>
      {err && <div style={{ color: '#a05a00', fontSize: 13, marginBottom: 8 }}>{err}</div>}

      <div style={{ ...LABEL, marginTop: 14 }}>Changes</div>
      {events.length === 0 ? (
        <div style={{ color: '#556070', fontSize: 13, padding: '6px 0' }}>
          No changes recorded yet.
        </div>
      ) : (
        <table style={{ borderCollapse: 'collapse', fontSize: 13, maxWidth: 420 }}>
          <thead>
            <tr style={{ textAlign: 'left', color: '#6b7785' }}>
              <th style={{ padding: '4px 24px 6px 0', fontWeight: 600 }}>Rows trained</th>
              <th style={{ padding: '4px 0 6px', fontWeight: 600 }}>New value</th>
            </tr>
          </thead>
          <tbody>
            {events
              .slice()
              .reverse()
              .map((e, i) => (
                <tr key={i} style={{ borderTop: '1px solid #e2e8f0' }}>
                  <td style={{ padding: '4px 24px 4px 0', fontFamily: 'monospace' }}>
                    {e.positions.toLocaleString()}
                  </td>
                  <td style={{ padding: '4px 0', fontFamily: 'monospace' }}>
                    {e.value.toExponential(3)}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
