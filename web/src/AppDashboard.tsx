import { useCallback, useEffect, useRef, useState } from 'react';
import BokehFigure from './components/BokehFigure';

// The React training dashboard. It renders the existing Bokeh metrics plots by
// embedding json_items fetched from the Python (Tornado) data API, with the shell
// state (tag selection, tabs, the loss-normalization toggle, polling) owned here in
// React. The engine injects VITE_TASK when it launches this dev server; a
// standalone run falls back to a ?task= query param. See docs/react_dashboard.md.
//
// Phase A scope: the tag selector, live polling, and the Training tab. The other
// tabs (positions, calibration, streaming) and the lane-analysis tab are added in
// later phases.

const DEFAULT_TASK = 'post_move_value';

function requestedTask(): string {
  const params = new URLSearchParams(window.location.search);
  return import.meta.env.VITE_TASK ?? params.get('task') ?? DEFAULT_TASK;
}

async function getJSON(url: string): Promise<any> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

export default function AppDashboard() {
  const task = requestedTask();
  const [tags, setTags] = useState<string[]>([]);
  const [tag, setTag] = useState<string | null>(null);
  const [normalized, setNormalized] = useState(false);
  const [item, setItem] = useState<unknown | null>(null);
  const lastVersion = useRef<number>(-1);

  // Tag list (refreshed every poll so a newly-started run appears).
  const refreshTags = useCallback(() => {
    getJSON(`/api/tags?task=${task}`)
      .then((d) => {
        setTags(d.tags);
        setTag((prev) => prev ?? d.tags[0] ?? null);
      })
      .catch(() => {});
  }, [task]);
  useEffect(refreshTags, [refreshTags]);

  const refetchFigure = useCallback(async () => {
    if (!tag) {
      setItem(null);
      return;
    }
    const url = `/api/figure/train_step?task=${task}&tag=${encodeURIComponent(tag)}&normalized=${
      normalized ? 1 : 0
    }`;
    const d = await getJSON(url);
    setItem(d.item ?? null);
  }, [task, tag, normalized]);

  // Re-fetch the figure on tag / toggle change.
  useEffect(() => {
    lastVersion.current = -1;
    refetchFigure().catch(() => {});
  }, [refetchFigure]);

  // Poll the cheap version token; re-fetch the figure only when the run advances.
  useEffect(() => {
    if (!tag) return;
    const id = setInterval(async () => {
      refreshTags();
      try {
        const v = await getJSON(`/api/version?task=${task}&tag=${encodeURIComponent(tag)}`);
        if (v.train_step !== lastVersion.current) {
          lastVersion.current = v.train_step;
          refetchFigure().catch(() => {});
        }
      } catch {
        /* tag may have no DB yet */
      }
    }, 3000);
    return () => clearInterval(id);
  }, [task, tag, refetchFigure, refreshTags]);

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: '12px 16px', color: '#223' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 12 }}>
        <strong style={{ fontSize: 16 }}>Scribblez dashboard</strong>
        <span style={{ color: '#889', fontSize: 13 }}>{task}</span>
        <label style={{ fontSize: 13 }}>
          Tag{' '}
          <select value={tag ?? ''} onChange={(e) => setTag(e.target.value || null)}>
            {tags.length === 0 && <option value="">(no runs)</option>}
            {tags.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div style={{ borderBottom: '2px solid #1f77b4', marginBottom: 12, fontSize: 14 }}>
        <span style={{ padding: '4px 10px', background: '#1f77b4', color: 'white', borderRadius: '6px 6px 0 0' }}>
          Training
        </span>
      </div>

      <div
        style={{
          background: 'white',
          border: '1px solid #e3e3ea',
          borderRadius: 8,
          padding: 12,
          boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
        }}
      >
        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
          {(['Absolute', '%'] as const).map((label, i) => {
            const active = (i === 1) === normalized;
            return (
              <button
                key={label}
                onClick={() => setNormalized(i === 1)}
                style={{
                  fontSize: 12,
                  padding: '3px 10px',
                  border: '1px solid #1f77b4',
                  borderRadius: 4,
                  cursor: 'pointer',
                  background: active ? '#1f77b4' : 'white',
                  color: active ? 'white' : '#1f77b4',
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
        {item ? (
          <BokehFigure item={item} />
        ) : (
          <div style={{ color: '#889', fontStyle: 'italic', padding: 20 }}>
            No streaming metrics recorded yet.
          </div>
        )}
      </div>
    </div>
  );
}
