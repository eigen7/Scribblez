import { useEffect, useState } from 'react';
import type { MouseEvent } from 'react';
import { getJSON } from '../lib/api';

// The leaf's value of one move at one known position over every analyzed
// generation (/api/deep_bot_leaf_eval/history), for the Positions tab's
// selected move. The slider's generation is marked; a generation that did not
// value the move breaks the line. Where the move has a truth (what a perfect
// leaf would value it at), a dashed line marks it: the level the leaf's line
// should approach.

export interface HistoryPoint {
  generation: number;
  value: number | null;
}

const WIDTH = 320;
const HEIGHT = 170;
const PAD = { left: 40, right: 12, top: 10, bottom: 22 };
const PLOT_W = WIDTH - PAD.left - PAD.right;
const PLOT_H = HEIGHT - PAD.top - PAD.bottom;
// The narrowest value range drawn, so a flat series is not magnified into noise.
const MIN_SPAN = 0.02;
const Y_TICKS = 4;

interface Scale {
  x: (generation: number) => number;
  y: (value: number) => number;
  yTicks: number[];
  xTicks: number[];
}

function scale(points: HistoryPoint[], truth: number | null): Scale {
  const values = points.flatMap((p) => (p.value == null ? [] : [p.value]));
  if (truth != null) values.push(truth);
  const gens = points.map((p) => p.generation);
  const g0 = gens[0];
  const g1 = Math.max(gens[gens.length - 1], g0 + 1);
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  const pad = Math.max((hi - lo) * 0.1, (MIN_SPAN - (hi - lo)) / 2, 0);
  lo -= pad;
  hi += pad;
  const yTicks = Array.from({ length: Y_TICKS }, (_, i) => lo + ((hi - lo) * i) / (Y_TICKS - 1));
  return {
    x: (g) => PAD.left + ((g - g0) / (g1 - g0)) * PLOT_W,
    y: (v) => PAD.top + (1 - (v - lo) / (hi - lo)) * PLOT_H,
    yTicks,
    xTicks: [...new Set([g0, gens[Math.floor(gens.length / 2)], gens[gens.length - 1]])],
  };
}

// The line's path, a new segment after each generation without a value.
function linePath(points: HistoryPoint[], s: Scale): string {
  let path = '';
  let open = false;
  for (const p of points) {
    if (p.value == null) {
      open = false;
      continue;
    }
    path += `${open ? 'L' : 'M'}${s.x(p.generation).toFixed(1)},${s.y(p.value).toFixed(1)}`;
    open = true;
  }
  return path;
}

// The point nearest the pointer's x.
function nearest(points: HistoryPoint[], s: Scale, px: number): HistoryPoint {
  return points.reduce((best, p) => (Math.abs(s.x(p.generation) - px) < Math.abs(s.x(best.generation) - px) ? p : best));
}

const pctText = (v: number | null) => (v == null ? 'not valued' : `${(v * 100).toFixed(1)}%`);

function Marker({ p, s, dashed }: { p: HistoryPoint; s: Scale; dashed: boolean }) {
  const x = s.x(p.generation);
  return (
    <g>
      <line x1={x} x2={x} y1={PAD.top} y2={PAD.top + PLOT_H} stroke="#8a96a8" strokeWidth={1}
        strokeDasharray={dashed ? '3 3' : undefined} />
      {p.value != null && (
        <circle cx={x} cy={s.y(p.value)} r={4} fill="#3a86d4" stroke="#f4f6f9" strokeWidth={2} />
      )}
    </g>
  );
}

function Axes({ s }: { s: Scale }) {
  return (
    <g fontSize={10} fill="#5b6778">
      {s.yTicks.map((v) => (
        <g key={`y${v}`}>
          <line x1={PAD.left} x2={PAD.left + PLOT_W} y1={s.y(v)} y2={s.y(v)} stroke="#e1e6ec" />
          <text x={PAD.left - 4} y={s.y(v) + 3} textAnchor="end">{pctText(v)}</text>
        </g>
      ))}
      {s.xTicks.map((g, i) => (
        // The end labels hang inward, so neither is clipped at the edge.
        <text key={`x${g}`} x={s.x(g)} y={HEIGHT - 6}
          textAnchor={i === 0 ? 'start' : i === s.xTicks.length - 1 ? 'end' : 'middle'}>{g}</text>
      ))}
    </g>
  );
}

const TRUTH_COLOR = '#c2410c';

function TruthLine({ s, truth }: { s: Scale; truth: number }) {
  return (
    <line x1={PAD.left} x2={PAD.left + PLOT_W} y1={s.y(truth)} y2={s.y(truth)} stroke={TRUTH_COLOR}
      strokeWidth={1.5} strokeDasharray="5 3" />
  );
}

export function HistoryChart({
  points, generation, truth = null,
}: {
  points: HistoryPoint[];
  generation: number;
  truth?: number | null;
}) {
  const [hover, setHover] = useState<HistoryPoint | null>(null);
  if (!points.some((p) => p.value != null)) return <div className="muted">No generation valued this move.</div>;
  const s = scale(points, truth);
  const current = points.find((p) => p.generation === generation);
  const onMove = (e: MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    setHover(nearest(points, s, PAD.left + ((e.clientX - box.left) / box.width) * PLOT_W));
  };
  const shown = hover ?? current;
  return (
    <>
      <div className="dbp-history-readout">
        {shown ? `generation ${shown.generation} · ${pctText(shown.value)}` : ' '}
        {truth != null && <span style={{ color: TRUTH_COLOR }}> · truth {pctText(truth)}</span>}
      </div>
      <svg width={WIDTH} height={HEIGHT} role="img" aria-label="leaf eval over generations">
        <Axes s={s} />
        {truth != null && <TruthLine s={s} truth={truth} />}
        <path d={linePath(points, s)} fill="none" stroke="#3a86d4" strokeWidth={2} strokeLinejoin="round" />
        {current && <Marker p={current} s={s} dashed={false} />}
        {hover && hover !== current && <Marker p={hover} s={s} dashed />}
        <rect x={PAD.left} y={PAD.top} width={PLOT_W} height={PLOT_H} fill="transparent"
          onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
      </svg>
    </>
  );
}

function useHistory(task: string, tag: string, position: number, move: string, newest: number | null) {
  // Keyed by what they are of, so another move's series is never drawn under
  // this move's title; a refetch for a new generation keeps the line meanwhile.
  const key = `${task}/${tag}/${position}/${move}`;
  const [loaded, setLoaded] = useState<{ key: string; points: HistoryPoint[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setError(null);
    const q = new URLSearchParams({ task, tag, position: String(position), move });
    getJSON(`/api/deep_bot_leaf_eval/history?${q}`)
      .then((d: { history: HistoryPoint[] }) => { if (!cancelled) setLoaded({ key, points: d.history }); })
      .catch((e: Error) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [key, task, tag, position, move, newest]);
  return { points: loaded?.key === key ? loaded.points : null, error };
}

export default function LeafEvalHistory({
  task, tag, position, move, display, generation, newest, truth,
}: {
  task: string;
  tag: string;
  position: number;
  move: string;
  display: string;
  generation: number;
  newest: number | null; // the newest analyzed generation; a new one refetches
  truth: number | null; // the move's truth, null without one
}) {
  const { points, error } = useHistory(task, tag, position, move, newest);
  return (
    <div className="dbp-panel dbp-history">
      <div className="dbp-panel-title">Leaf eval of <span className="dbp-answer-move">{display}</span> by generation</div>
      {error && <div className="dbp-error">{error}</div>}
      {!error && (points ? <HistoryChart points={points} generation={generation} truth={truth} /> : (
        <div className="muted"><span className="scz-spinner" style={{ verticalAlign: 'middle' }} /> reading every generation…</div>
      ))}
    </div>
  );
}
