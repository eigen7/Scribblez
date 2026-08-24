import { useContext, useCallback, useEffect, useState } from 'react';
import { TabActiveContext } from './TabActiveContext';
import { getJSON, postJSON } from '../lib/api';

// The Controls tab: live operator knobs read from and written to the per-tag
// dashboard.db via /api/controls. The generational trainer adopts changes at its
// next generation and records a rows-clock change event for each; the history
// table below also shows the LR schedule's phase-boundary events. Setting a value
// here persists across trainer restarts.

type ControlEvent = { positions: number; name: string; value: number; t: number };
type ControlsData = { controls: Record<string, number>; events: ControlEvent[] };

type Spec = { name: string; label: string; hint: string; fmt: (v: number) => string };

const LABEL: React.CSSProperties = {
  fontSize: 12,
  color: '#6b7785',
  textTransform: 'uppercase',
  letterSpacing: 0.4,
};
const EMPTY: React.CSSProperties = { color: '#556070', fontStyle: 'italic', padding: 20 };

const int = (v: number) => String(Math.round(v));
// 'lr' events are the schedule's phase boundaries; 'base_lr' is the retired manual
// control, still present in older tags' history.
const isLrEvent = (name: string) => name === 'lr' || name === 'base_lr';
const CONTROLS: Spec[] = [
  { name: 'dataloader_workers', label: 'DataLoader workers', hint: 'C++ decode/shuffle workers, applied at the next generation.', fmt: int },
  { name: 'torch_threads', label: 'PyTorch intra-op threads', hint: 'Applied at the next generation.', fmt: int },
];

function NumControl({
  url,
  spec,
  value,
  refetch,
}: {
  url: string;
  spec: Spec;
  value: number | undefined;
  refetch: () => Promise<void>;
}) {
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const submit = useCallback(async () => {
    const v = Number(draft);
    if (!draft.trim() || !Number.isFinite(v) || v <= 0) {
      setErr('Enter a positive number.');
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      await postJSON(url, { name: spec.name, value: v });
      setDraft('');
      await refetch();
    } catch (e) {
      setErr(String(e));
    }
    setBusy(false);
  }, [url, spec.name, draft, refetch]);

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={LABEL}>{spec.label}</div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <span style={{ fontFamily: 'monospace', fontSize: 16, minWidth: 80 }}>
          {value != null ? spec.fmt(value) : '—'}
        </span>
        <input
          type="text"
          value={draft}
          placeholder={value != null ? spec.fmt(value) : ''}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') submit();
          }}
          style={{ fontFamily: 'monospace', width: 110, padding: '4px 8px' }}
        />
        <button onClick={submit} disabled={busy}>
          Set
        </button>
        {err && <span style={{ color: '#a05a00', fontSize: 12 }}>{err}</span>}
      </div>
      <div style={{ fontSize: 12, color: '#6b7785', marginTop: 2 }}>{spec.hint}</div>
    </div>
  );
}

export default function ControlsTab({ task, tag }: { task: string; tag: string | null }) {
  const [data, setData] = useState<ControlsData | null>(null);
  const tabActive = useContext(TabActiveContext);
  const [loaded, setLoaded] = useState(false);

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

  // Refetch on run switch, and poll (the draft inputs are separate state, so
  // polling never clobbers what the operator is typing).
  useEffect(() => {
    setLoaded(false);
    refetch().catch(() => {});
  }, [refetch]);
  useEffect(() => {
    if (!tabActive) return;
    const id = setInterval(() => refetch().catch(() => {}), 5000);
    return () => clearInterval(id);
  }, [refetch, tabActive]);

  if (!tag || !url) {
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

  const controls = data?.controls ?? {};
  const events = data?.events ?? [];

  return (
    <div className="card">
      {CONTROLS.map((spec) => (
        <NumControl key={spec.name} url={url} spec={spec} value={controls[spec.name]} refetch={refetch} />
      ))}

      <div style={{ ...LABEL, marginTop: 10 }}>Changes</div>
      {events.length === 0 ? (
        <div style={{ color: '#556070', fontSize: 13, padding: '6px 0' }}>
          No changes recorded yet.
        </div>
      ) : (
        <table style={{ borderCollapse: 'collapse', fontSize: 13, maxWidth: 520 }}>
          <thead>
            <tr style={{ textAlign: 'left', color: '#6b7785' }}>
              <th style={{ padding: '4px 24px 6px 0', fontWeight: 600 }}>Rows trained</th>
              <th style={{ padding: '4px 24px 6px 0', fontWeight: 600 }}>Control</th>
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
                  <td style={{ padding: '4px 24px 4px 0', fontFamily: 'monospace' }}>{e.name}</td>
                  <td style={{ padding: '4px 0', fontFamily: 'monospace' }}>
                    {isLrEvent(e.name) ? e.value.toExponential(3) : int(e.value)}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
