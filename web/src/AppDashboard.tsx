import { Component, ReactNode, useCallback, useEffect, useRef, useState } from 'react';
import BokehFigure from './components/BokehFigure';
import ControlsTab from './components/ControlsTab';
import InfoTab from './components/InfoTab';
import LaneAnalysis from './components/LaneAnalysis';
import MasterApp from './components/master/MasterApp';
import PostMoveAnalysis from './components/PostMoveAnalysis';
import { getJSON } from './lib/api';

// Contains a render error to the active tab: a crashing tab shows an inline message
// instead of unmounting (blanking) the whole dashboard. Reset by remounting on a
// `key` change (tab switch), so navigating away and back retries the tab.
class TabErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="card" style={{ color: '#a05a00', padding: 16 }}>
          <b>This tab hit an error.</b>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12, marginTop: 8 }}>
            {String(this.state.error.message || this.state.error)}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

// The dashboard app has two faces. With an explicit training task (VITE_TASK,
// injected when a trainer launches the dev server, or a ?task= query param) it
// renders that task's training dashboard: the Bokeh metrics plots embedded as
// json_items fetched from the Python (Tornado) data API, with shell state (tag
// selection, tabs, polling) owned in React (docs/react_dashboard.md). With no
// task it renders the master dashboard -- the entrypoint for all work
// (docs/master_dashboard.md).

const MAX_MOVE_PER_LANE = 'max_move_per_lane';

function requestedTask(): string | null {
  const params = new URLSearchParams(window.location.search);
  return import.meta.env.VITE_TASK ?? params.get('task');
}

// A `?tag=<tag>` query param opens the dashboard on that run (the trainer prints
// such a URL); absent, the tag selector defaults to the first available tag.
function requestedTag(): string | null {
  return new URLSearchParams(window.location.search).get('tag');
}

// A segmented single-select control (connected buttons, exactly one active) --
// the Loss tab's Absolute/% selector.
function RadioButtonGroup({ options, value, onChange }: {
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
function useFigureItem(
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
function FigureBody({ item, emptyText }: { item: unknown | null; emptyText: string }) {
  return item ? (
    <BokehFigure item={item} />
  ) : (
    <div style={{ color: '#556070', fontStyle: 'italic', padding: 20 }}>{emptyText}</div>
  );
}

// A tab that embeds a single Bokeh figure with no controls (Training).
function FigureTab({
  task, tag, figure, versionKey, emptyText,
}: {
  task: string; tag: string | null; figure: string; versionKey: string | string[]; emptyText: string;
}) {
  const item = useFigureItem(task, tag, figure, versionKey, '');
  return <div className="card"><FigureBody item={item} emptyText={emptyText} /></div>;
}

// The Loss tab: the loss/accuracy figure on top (with an Absolute/% selector), then
// the value-quality figure below, with the controls that govern it -- Smooth and a
// Secondary tag to overlay -- sitting between the two. The two figures are fetched
// separately so those controls can live between them.
const LOSS_VERSION = ['metrics', 'control_event'];
const QUALITY_VERSION = ['metrics'];

function LossTab({ task, tag, tags }: { task: string; tag: string | null; tags: string[] }) {
  const [normalized, setNormalized] = useState(false);
  const [smoothed, setSmoothed] = useState(true); // smoothing on by default
  const [secondary, setSecondary] = useState(''); // '' = none

  const stepItem = useFigureItem(
    task, tag, 'loss', LOSS_VERSION, `&normalized=${normalized ? 1 : 0}`,
  );
  const qualityItem = useFigureItem(
    task, tag, 'eval_quality', QUALITY_VERSION,
    `&smooth=${smoothed ? 1 : 0}${secondary ? `&secondary=${encodeURIComponent(secondary)}` : ''}`,
  );
  const otherTags = tags.filter((t) => t !== tag);

  return (
    <div className="card">
      <div style={{ marginBottom: 8 }}>
        <RadioButtonGroup
          options={['Absolute', '%']}
          value={normalized ? 1 : 0}
          onChange={(i) => setNormalized(i === 1)}
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

// Non-Loss embedded-figure tabs: which API figure to fetch, which version-token
// table to poll for changes, and the empty message. ("Lane analysis" and the Loss
// tab are handled separately.)
const FIGURE_TABS: Record<
  string,
  { figure: string; versionKey: string | string[]; emptyText: string }
> = {
  Training: { figure: 'training_metrics', versionKey: 'metrics',
    emptyText: 'No per-epoch training metrics yet.' },
};

function renderTab(name: string, task: string, tag: string | null, tags: string[]) {
  if (name === 'Info') return <InfoTab task={task} tag={tag} />;
  if (name === 'Controls') return <ControlsTab task={task} tag={tag} />;
  if (name === 'Lane analysis') return <LaneAnalysis task={task} tag={tag} />;
  if (name === 'Positions') return <PostMoveAnalysis task={task} tag={tag} />;
  if (name === 'Loss') return <LossTab task={task} tag={tag} tags={tags} />;
  const cfg = FIGURE_TABS[name];
  return (
    <FigureTab
      task={task} tag={tag} figure={cfg.figure} versionKey={cfg.versionKey} emptyText={cfg.emptyText}
    />
  );
}

export default function AppDashboard() {
  const task = requestedTask();
  return task ? <TrainingDashboard task={task} /> : <MasterApp />;
}

function TrainingDashboard({ task }: { task: string }) {
  const tabs =
    task === MAX_MOVE_PER_LANE
      ? ['Loss', 'Lane analysis', 'Info']
      : ['Loss', 'Positions', 'Training', 'Controls', 'Info'];
  const [tags, setTags] = useState<string[]>([]);
  const [tag, setTag] = useState<string | null>(requestedTag());
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

  // Keep the `?tag=<tag>` query param in sync with the selected tag (whether the
  // user picked it from the dropdown or it defaulted to the first run), so the URL
  // is shareable and reload-stable. replaceState (not pushState) avoids stacking a
  // history entry per selection; other params (e.g. `task`) are preserved.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (tag) params.set('tag', tag);
    else params.delete('tag');
    const query = params.toString();
    window.history.replaceState(null, '', `${window.location.pathname}${query ? `?${query}` : ''}`);
  }, [tag]);

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

      <TabErrorBoundary key={tabs[tab]}>{renderTab(tabs[tab], task, tag, tags)}</TabErrorBoundary>
    </div>
  );
}
