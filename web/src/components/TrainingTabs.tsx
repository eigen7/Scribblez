import { MutableRefObject, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getJSON, postJSON } from '../lib/api';
import { FigureHandle } from '../lib/bokehDoc';
import { Visibility } from '../lib/bokehVisibility';
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

// Fetch one Bokeh figure (a json_item) for the given tag and keep it current as
// the run advances (polling a cheap version token). A full re-fetch happens on
// mount and whenever `query` changes (a control was toggled); a version bump is
// instead applied incrementally when `handleRef` exposes an embedded document --
// the appended rows are streamed into it in place (lib/bokehDoc.ts,
// figure_delta.py) -- falling back to the full fetch whenever the delta endpoint
// asks for it or fails.
export function useFigureItem(
  task: string, tag: string | null, figure: string,
  versionKey: string | string[], query: string,
  handleRef?: MutableRefObject<FigureHandle | null>,
): unknown | null {
  const [item, setItem] = useState<unknown | null>(null);
  const lastVersion = useRef<number>(-1);
  const structureRef = useRef<string | null>(null); // the embedded doc's structure key

  const refetch = useCallback(async () => {
    if (!tag) {
      setItem(null);
      structureRef.current = null;
      return;
    }
    const d = await getJSON(
      `/api/figure/${figure}?task=${task}&tag=${encodeURIComponent(tag)}${query}`,
    );
    structureRef.current = d.structure ?? null;
    setItem(d.item ?? null);
  }, [task, tag, figure, query]);

  const advance = useCallback(async () => {
    const handle = handleRef?.current;
    const sources = handle && structureRef.current ? handle.sourceStates() : null;
    if (handle && sources) {
      try {
        const d = await postJSON(
          `/api/figure_delta/${figure}?task=${task}&tag=${encodeURIComponent(tag ?? '')}${query}`,
          { structure: structureRef.current, sources },
        );
        if (!d.refetch) {
          handle.applyDelta(d);
          return;
        }
      } catch {
        /* fall through to the full fetch */
      }
    }
    await refetch();
  }, [task, tag, figure, query, handleRef, refetch]);

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
          advance().catch(() => {});
        }
      } catch {
        /* the API may still be starting (heavy torch import); the poll retries */
      }
    }, 3000);
    return () => clearInterval(id);
  }, [task, tag, versionKey, advance]);

  return item;
}

// The embedded figure, or an italic placeholder when the API has no data for it.
// `visibility` drives named models inside it; `handleRef` exposes it to the
// incremental-refresh path (see BokehFigure).
export function FigureBody({ item, emptyText, visibility, handleRef }: {
  item: unknown | null; emptyText: string; visibility?: Visibility;
  handleRef?: MutableRefObject<FigureHandle | null>;
}) {
  return item ? (
    <BokehFigure item={item} visibility={visibility} handleRef={handleRef} />
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
//
// Linear x/Log x never talks to the API: each figure arrives with both x-axis
// variants as two named rows (plots.py's X_AXIS_LINEAR / X_AXIS_LOG -- change the
// names in both places), and the knob flips their visibility in place.
const LOSS_VERSION = ['metrics', 'control_event'];
const QUALITY_VERSION = ['metrics'];
const X_AXIS_LINEAR = 'x_linear';
const X_AXIS_LOG = 'x_log';

export function LossTab({ task, tag }: { task: string; tag: string | null }) {
  const [normalized, setNormalized] = useState(false);
  const lossHandle = useRef<FigureHandle | null>(null);
  const qualityHandle = useRef<FigureHandle | null>(null);
  const [logX, setLogX] = useState(false);
  const [smoothed, setSmoothed] = useState(true); // smoothing on by default
  const [secondary, setSecondary] = useState(''); // '' = none
  const [tags, setTags] = useState<string[]>([]); // secondary-overlay choices

  useEffect(() => {
    getJSON(`/api/tags?task=${task}`).then((d) => setTags(d.tags)).catch(() => {});
  }, [task]);

  const stepItem = useFigureItem(
    task, tag, 'loss', LOSS_VERSION, `&normalized=${normalized ? 1 : 0}`, lossHandle,
  );
  const qualityItem = useFigureItem(
    task, tag, 'eval_quality', QUALITY_VERSION,
    `&smooth=${smoothed ? 1 : 0}${secondary ? `&secondary=${encodeURIComponent(secondary)}` : ''}`,
    qualityHandle,
  );
  const otherTags = tags.filter((t) => t !== tag);
  const xAxisRows = useMemo<Visibility>(
    () => ({ [X_AXIS_LINEAR]: !logX, [X_AXIS_LOG]: logX }), [logX],
  );

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
      <FigureBody
        item={stepItem} emptyText="No loss / accuracy metrics recorded yet." visibility={xAxisRows}
        handleRef={lossHandle}
      />

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
          <FigureBody
            item={qualityItem} emptyText="" visibility={xAxisRows} handleRef={qualityHandle}
          />
        </>
      )}
    </div>
  );
}
