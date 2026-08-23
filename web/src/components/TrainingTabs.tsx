import { useCallback, useEffect, useRef, useState } from 'react';
import { getJSON } from '../lib/api';
import BokehFigure from './BokehFigure';

// The embedded-Bokeh training tabs (Loss, Training) shared by the training
// workloads' task views: each fetches a figure as a json_item from the Python
// data API and re-fetches when the run advances (polling a cheap version
// token) or a control changes.

// A segmented single-select control (connected buttons, exactly one active) --
// the Loss tab's Absolute/% and Linear x/Log x selectors.
export function RadioButtonGroup({ options, value, onChange }: {
  options: readonly string[]; value: number; onChange: (i: number) => void;
}) {
  return (
    <div style={{ display: 'inline-flex', border: '1px solid #1f77b4', borderRadius: 4, overflow: 'hidden' }}>
      {options.map((label, i) => (
        <button
          key={label}
          onClick={() => onChange(i)}
          style={{
            fontSize: 12, padding: '3px 12px', border: 'none',
            borderLeft: i === 0 ? 'none' : '1px solid #1f77b4', cursor: 'pointer',
            background: i === value ? '#1f77b4' : 'white', color: i === value ? 'white' : '#1f77b4',
          }}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

// Fetch one Bokeh figure (a json_item) for the given tag and re-fetch it whenever
// the run advances (polling a cheap version token) or `query` changes (a control
// was toggled). `query` is extra query string appended to the figure request.
export function useFigureItem(
  task: string, tag: string | null, figure: string,
  versionKey: string | string[], query: string,
): unknown | null {
  const [item, setItem] = useState<unknown | null>(null);
  const lastVersion = useRef<number>(-1);

  const refetch = useCallback(async () => {
    if (!tag) {
      setItem(null);
      return;
    }
    const d = await getJSON(
      `/api/figure/${figure}?task=${task}&tag=${encodeURIComponent(tag)}${query}`,
    );
    setItem(d.item ?? null);
  }, [task, tag, figure, query]);

  useEffect(() => {
    lastVersion.current = -1;
    refetch().catch(() => {});
  }, [refetch]);

  useEffect(() => {
    if (!tag) return;
    const id = setInterval(async () => {
      try {
        const v = await getJSON(`/api/version?task=${task}&tag=${encodeURIComponent(tag)}`);
        const keys = Array.isArray(versionKey) ? versionKey : [versionKey];
        const combined = keys.reduce((sum, k) => sum + (v[k] ?? 0), 0);
        if (combined !== lastVersion.current) {
          lastVersion.current = combined;
          refetch().catch(() => {});
        }
      } catch {
        /* the API may still be starting (heavy torch import); the poll retries */
      }
    }, 3000);
    return () => clearInterval(id);
  }, [task, tag, versionKey, refetch]);

  return item;
}

// The embedded figure, or an italic placeholder when the API has no data for it.
export function FigureBody({ item, emptyText }: { item: unknown | null; emptyText: string }) {
  return item ? (
    <BokehFigure item={item} />
  ) : (
    <div style={{ color: '#556070', fontStyle: 'italic', padding: 20 }}>{emptyText}</div>
  );
}

// A tab that embeds a single Bokeh figure with no controls (Training).
export function FigureTab({
  task, tag, figure, versionKey, emptyText,
}: {
  task: string; tag: string | null; figure: string; versionKey: string | string[]; emptyText: string;
}) {
  const item = useFigureItem(task, tag, figure, versionKey, '');
  return <div className="card"><FigureBody item={item} emptyText={emptyText} /></div>;
}

// The Loss tab: the loss/accuracy figure on top (with an Absolute/% selector and
// the Linear x/Log x selector that governs the x-axis of BOTH figures), then the
// value-quality figure below, with the controls specific to it -- Smooth and a
// Secondary tag to overlay -- sitting between the two. The two figures are
// fetched separately so those controls can live between them.
const LOSS_VERSION = ['metrics', 'control_event'];
const QUALITY_VERSION = ['metrics'];

export function LossTab({ task, tag }: { task: string; tag: string | null }) {
  const [normalized, setNormalized] = useState(false);
  const [logX, setLogX] = useState(false);
  const [smoothed, setSmoothed] = useState(true); // smoothing on by default
  const [secondary, setSecondary] = useState(''); // '' = none
  const [tags, setTags] = useState<string[]>([]); // secondary-overlay choices

  useEffect(() => {
    getJSON(`/api/tags?task=${task}`).then((d) => setTags(d.tags)).catch(() => {});
  }, [task]);

  const stepItem = useFigureItem(
    task, tag, 'loss', LOSS_VERSION, `&normalized=${normalized ? 1 : 0}&log_x=${logX ? 1 : 0}`,
  );
  const qualityItem = useFigureItem(
    task, tag, 'eval_quality', QUALITY_VERSION,
    `&smooth=${smoothed ? 1 : 0}&log_x=${logX ? 1 : 0}`
      + (secondary ? `&secondary=${encodeURIComponent(secondary)}` : ''),
  );
  const otherTags = tags.filter((t) => t !== tag);

  return (
    <div className="card">
      <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
        <RadioButtonGroup
          options={['Absolute', '%']}
          value={normalized ? 1 : 0}
          onChange={(i) => setNormalized(i === 1)}
        />
        <RadioButtonGroup
          options={['Linear x', 'Log x']}
          value={logX ? 1 : 0}
          onChange={(i) => setLogX(i === 1)}
        />
      </div>
      <FigureBody item={stepItem} emptyText="No loss / accuracy metrics recorded yet." />

      {/* Controls + figure appear only once value-quality curves exist (they are
          absent for tasks/runs without a Monte-Carlo quality eval). */}
      {qualityItem != null && (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: 18, margin: '16px 0 8px' }}>
            <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 5, cursor: 'pointer' }}>
              <input type="checkbox" checked={smoothed} onChange={(e) => setSmoothed(e.target.checked)} />
              Smooth
            </label>
            <label style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
              Secondary tag:
              <select value={secondary} onChange={(e) => setSecondary(e.target.value)}>
                <option value="">(none)</option>
                {otherTags.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </label>
          </div>
          <FigureBody item={qualityItem} emptyText="" />
        </>
      )}
    </div>
  );
}
