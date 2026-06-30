import { useCallback, useEffect, useRef, useState } from 'react';
import BokehFigure from './components/BokehFigure';
import LaneAnalysis from './components/LaneAnalysis';
import { getJSON } from './lib/api';

// The React training dashboard. It renders the existing Bokeh metrics plots by
// embedding json_items fetched from the Python (Tornado) data API, with the shell
// state (tag selection, tabs, the loss-normalization toggle, polling) owned here in
// React. The engine injects VITE_TASK when it launches this dev server; a
// standalone run falls back to a ?task= query param. See docs/react_dashboard.md.

const DEFAULT_TASK = 'post_move_value';
const MAX_MOVE_PER_LANE = 'max_move_per_lane';

function requestedTask(): string {
  const params = new URLSearchParams(window.location.search);
  return import.meta.env.VITE_TASK ?? params.get('task') ?? DEFAULT_TASK;
}

// A tab that embeds one Bokeh figure (fetched as a json_item from the data API)
// and re-fetches it whenever the run advances (polling a cheap version token). The
// loss figure adds an Absolute/% toggle; others (throughput) pass `toggle={false}`.
function FigureTab({
  task, tag, figure, versionKey, toggle, emptyText,
}: {
  task: string; tag: string | null; figure: string; versionKey: string;
  toggle: boolean; emptyText: string;
}) {
  const [normalized, setNormalized] = useState(false);
  const [item, setItem] = useState<unknown | null>(null);
  const lastVersion = useRef<number>(-1);

  const refetch = useCallback(async () => {
    if (!tag) {
      setItem(null);
      return;
    }
    const q = toggle ? `&normalized=${normalized ? 1 : 0}` : '';
    const d = await getJSON(`/api/figure/${figure}?task=${task}&tag=${encodeURIComponent(tag)}${q}`);
    setItem(d.item ?? null);
  }, [task, tag, figure, toggle, normalized]);

  useEffect(() => {
    lastVersion.current = -1;
    refetch().catch(() => {});
  }, [refetch]);

  useEffect(() => {
    if (!tag) return;
    const id = setInterval(async () => {
      try {
        const v = await getJSON(`/api/version?task=${task}&tag=${encodeURIComponent(tag)}`);
        if (v[versionKey] !== lastVersion.current) {
          lastVersion.current = v[versionKey];
          refetch().catch(() => {});
        }
      } catch {
        /* the API may still be starting (heavy torch import); the poll retries */
      }
    }, 3000);
    return () => clearInterval(id);
  }, [task, tag, versionKey, refetch]);

  return (
    <div className="card">
      {toggle && (
        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
          {(['Absolute', '%'] as const).map((label, i) => {
            const active = (i === 1) === normalized;
            return (
              <button
                key={label}
                onClick={() => setNormalized(i === 1)}
                style={{
                  fontSize: 12, padding: '3px 10px', border: '1px solid #1f77b4', borderRadius: 4,
                  cursor: 'pointer', background: active ? '#1f77b4' : 'white', color: active ? 'white' : '#1f77b4',
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      )}
      {item ? (
        <BokehFigure item={item} />
      ) : (
        <div style={{ color: '#556070', fontStyle: 'italic', padding: 20 }}>{emptyText}</div>
      )}
    </div>
  );
}

export default function AppDashboard() {
  const task = requestedTask();
  const tabs =
    task === MAX_MOVE_PER_LANE
      ? ['Training', 'Performance', 'Lane analysis']
      : ['Training', 'Performance'];
  const [tags, setTags] = useState<string[]>([]);
  const [tag, setTag] = useState<string | null>(null);
  const [tab, setTab] = useState(0);

  const refreshTags = useCallback(() => {
    getJSON(`/api/tags?task=${task}`)
      .then((d) => {
        setTags(d.tags);
        setTag((prev) => prev ?? d.tags[0] ?? null);
      })
      .catch(() => {});
  }, [task]);
  useEffect(refreshTags, [refreshTags]);
  useEffect(() => {
    const id = setInterval(refreshTags, 3000);
    return () => clearInterval(id);
  }, [refreshTags]);

  return (
    <div style={{
      fontFamily: 'system-ui, sans-serif', padding: '14px 18px', color: '#1a1f28', fontSize: 15,
      background: '#f4f6f8', minHeight: '100vh',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 14 }}>
        <strong style={{ fontSize: 19 }}>Scribblez dashboard</strong>
        <span style={{ color: '#445063', fontSize: 14 }}>{task}</span>
        <label style={{ fontSize: 15 }}>
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

      {tabs[tab] === 'Lane analysis' ? (
        <LaneAnalysis task={task} tag={tag} />
      ) : tabs[tab] === 'Performance' ? (
        <FigureTab
          task={task} tag={tag} figure="throughput" versionKey="throughput"
          toggle={false} emptyText="No throughput data yet — start a streaming run."
        />
      ) : (
        <FigureTab
          task={task} tag={tag} figure="train_step" versionKey="train_step"
          toggle emptyText="No streaming metrics recorded yet."
        />
      )}
    </div>
  );
}
