import { useEffect, useMemo, useState } from 'react';
import Board from './Board';
import GenerationSlider from './GenerationSlider';
import Rack from './Rack';
import UnseenTiles from './UnseenTiles';
import { TileInfo } from '../types';
import { getJSON } from '../lib/api';
import {
  buildPlacementOverlay,
  overlayGradient,
  HEAD_OPTIONS,
  MODEL_HUE,
  SIM_HUE,
  FLOOR,
  RESIDUAL_CAP,
  RAW_CAP,
  OverlayMode,
  PlacementData,
  PlacementHeadData,
  PlacementHeadKey,
} from '../lib/placementOverlay';

const NO_USED: Set<number> = new Set();
const MC_COLOR = '#e74c3c'; // Monte-Carlo ground truth (red)
const MODEL_COLOR = '#3a86d4'; // model prediction (blue)

interface MC {
  n: number;
  wld: { win: number; loss: number; draw: number };
  score_delta_hist: [number, number][]; // sorted [delta, count]
  score_delta_mean: number;
}

// The standard deviation of the exact MC score-delta distribution, derived from its
// histogram (count-weighted about the mean) so it needs no extra API field.
function mcStd(mc: MC): number {
  const total = mc.score_delta_hist.reduce((s, [, c]) => s + c, 0) || 1;
  const v = mc.score_delta_hist.reduce((s, [d, c]) => s + c * (d - mc.score_delta_mean) ** 2, 0);
  return Math.sqrt(v / total);
}

interface Model {
  wld: { win: number; draw: number; loss: number };
  sd_mean: number;
  sd_std: number;
}

interface AltResult {
  leave: string;
  generation: number;
  model: Model;
}

// The model's evaluation of a hypothetical alternate leave (model only -- there is no
// Monte-Carlo ground truth for an arbitrary leave).
function AltLeaveResult({ result }: { result: AltResult }) {
  const m = result.model;
  return (
    <div style={{ marginTop: 8, padding: '8px 11px', background: '#eef3f8', borderRadius: 6, fontSize: 13, maxWidth: 360 }}>
      <div>
        Alternate leave <b>{result.leave.toUpperCase()}</b>{' '}
        <span style={{ color: '#667', fontSize: 12 }}>(model only — gen {result.generation})</span>
      </div>
      <div style={{ marginTop: 4 }}>
        Win <b style={{ color: MODEL_COLOR }}>{(m.wld.win * 100).toFixed(1)}%</b> · Loss{' '}
        <b>{(m.wld.loss * 100).toFixed(1)}%</b> · Draw <b>{(m.wld.draw * 100).toFixed(1)}%</b>
      </div>
      <div>
        Score-delta μ <b>{m.sd_mean.toFixed(1)}</b> (σ {m.sd_std.toFixed(1)})
      </div>
    </div>
  );
}

interface Payload {
  name: string;
  start_player: number;
  last_move: [number, number][];
  board: (string | null)[][];
  bonuses: (string | null)[][];
  rack: TileInfo[];
  tile_scores: Record<string, number>;
  scores: [number, number];
  bag_count: number;
  opponent_rack_count: number;
  generation: number | null;
  has_prediction: boolean;
  mc: MC;
  model: Model | null;
  // Ground-truth vs. predicted per-square probabilities for the model's four
  // placement heads, absent (or null) when the backend hasn't computed them
  // for this position/generation yet.
  placement?: PlacementData | null;
}

interface Generation {
  generation: number;
  positions: number;
}

// One model-vs-MC pair of bars for a win/loss/draw outcome.
function WldGroup({ label, mc, model }: { label: string; mc: number; model: number | null }) {
  const H = 120;
  const bar = (value: number, color: string) => (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 34 }}>
      <span style={{ fontSize: 11, color: '#445063' }}>{(value * 100).toFixed(0)}%</span>
      <div
        style={{ width: 22, height: Math.max(1, value * H), background: color, borderRadius: '3px 3px 0 0' }}
        title={`${(value * 100).toFixed(1)}%`}
      />
    </div>
  );
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: H }}>
        {model != null && bar(model, MODEL_COLOR)}
        {bar(mc, MC_COLOR)}
      </div>
      <div style={{ fontWeight: 600, fontSize: 13 }}>{label}</div>
    </div>
  );
}

function WldChart({ mc, model }: { mc: MC; model: Model | null }) {
  return (
    <div>
      <div style={{ display: 'flex', gap: 30, alignItems: 'flex-end', padding: '4px 8px' }}>
        <WldGroup label="Win" mc={mc.wld.win} model={model?.wld.win ?? null} />
        <WldGroup label="Loss" mc={mc.wld.loss} model={model?.wld.loss ?? null} />
        <WldGroup label="Draw" mc={mc.wld.draw} model={model?.wld.draw ?? null} />
      </div>
      <div style={{ fontSize: 12, color: '#445063', display: 'flex', gap: 14, paddingLeft: 8 }}>
        <span><span style={{ ...SW, background: MODEL_COLOR }} /> model</span>
        <span><span style={{ ...SW, background: MC_COLOR }} /> Monte-Carlo ({mc.n.toLocaleString()} games)</span>
      </div>
    </div>
  );
}

const SW: React.CSSProperties = {
  display: 'inline-block', width: 11, height: 11, borderRadius: 2, marginRight: 4, verticalAlign: 'middle',
};

// Bin the exact (delta -> count) histogram into `nbins` buckets over [lo, hi),
// returning the count in each bucket.
function binCounts(hist: [number, number][], lo: number, hi: number, nbins: number): number[] {
  const w = (hi - lo) / nbins;
  const out = new Array(nbins).fill(0);
  for (const [d, c] of hist) {
    let b = Math.floor((d - lo) / w);
    if (b < 0) b = 0;
    if (b >= nbins) b = nbins - 1;
    out[b] += c;
  }
  return out;
}

// The MC final-score-delta distribution (binned bars) with the model's predicted
// Gaussian (mean/std) overlaid. Both are shown as probability mass per bucket, so
// the curve and the bars are directly comparable.
function ScoreDeltaChart({ mc, model }: { mc: MC; model: Model | null }) {
  const hist = mc.score_delta_hist;
  if (hist.length === 0) return <div className="muted" style={{ fontStyle: 'italic' }}>No ground truth.</div>;

  const W = 640, H = 220, padL = 30, padR = 12, padT = 12, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const nbins = 48;

  const mcMean = mc.score_delta_mean, mcSig = mcStd(mc);
  let lo = hist[0][0], hi = hist[hist.length - 1][0];
  lo = Math.min(lo, mcMean - 3 * mcSig);
  hi = Math.max(hi, mcMean + 3 * mcSig);
  if (model) {
    lo = Math.min(lo, model.sd_mean - 3 * model.sd_std);
    hi = Math.max(hi, model.sd_mean + 3 * model.sd_std);
  }
  const span = Math.max(1, hi - lo);
  lo -= span * 0.02;
  hi += span * 0.02;
  const w = (hi - lo) / nbins;

  const counts = binCounts(hist, lo, hi, nbins);
  const barMass = counts.map((c) => c / Math.max(1, mc.n));
  // A Gaussian's probability mass in one bucket (pdf x bucket width), so the curves
  // are on the same scale as the histogram bars.
  const gaussMass = (x: number, mean: number, std: number) =>
    (Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI))) * w;
  const peakMass = (mean: number, std: number) =>
    Math.max(...Array.from({ length: nbins }, (_, i) => gaussMass(lo + (i + 0.5) * w, mean, std)));
  const yMax = Math.max(
    ...barMass,
    peakMass(mcMean, mcSig),
    model ? peakMass(model.sd_mean, model.sd_std) : 0,
    1e-9,
  );

  const xPx = (x: number) => padL + ((x - lo) / (hi - lo)) * plotW;
  const yPx = (m: number) => padT + plotH - (m / yMax) * plotH;
  const curve = (mean: number, std: number) =>
    Array.from({ length: 121 }, (_, i) => {
      const x = lo + (i / 120) * (hi - lo);
      return `${xPx(x).toFixed(1)},${yPx(gaussMass(x, mean, std)).toFixed(1)}`;
    }).join(' ');
  const ticks = [lo, lo + (hi - lo) / 2, hi].map(Math.round);

  return (
    <svg width={W} height={H} style={{ maxWidth: '100%', height: 'auto' }}>
      {/* MC histogram (the exact distribution) as faint context behind the curves */}
      {barMass.map((m, i) => {
        const x0 = xPx(lo + i * w);
        const bw = Math.max(0.5, xPx(lo + (i + 1) * w) - x0 - 0.5);
        return <rect key={i} x={x0} y={yPx(m)} width={bw} height={padT + plotH - yPx(m)} fill={MC_COLOR} opacity={0.18} />;
      })}
      {/* Dashed vertical line at each Gaussian's mean, rising from the axis to that
          Gaussian's peak (same color as its curve). */}
      <line x1={xPx(mcMean)} y1={yPx(gaussMass(mcMean, mcMean, mcSig))} x2={xPx(mcMean)} y2={padT + plotH} stroke={MC_COLOR} strokeWidth={1.5} strokeDasharray="4 3" />
      {model && (
        <line x1={xPx(model.sd_mean)} y1={yPx(gaussMass(model.sd_mean, model.sd_mean, model.sd_std))} x2={xPx(model.sd_mean)} y2={padT + plotH} stroke={MODEL_COLOR} strokeWidth={1.5} strokeDasharray="4 3" />
      )}
      {/* MC Gaussian (red) and model Gaussian (blue) */}
      <polyline points={curve(mcMean, mcSig)} fill="none" stroke={MC_COLOR} strokeWidth={2} />
      {model && <polyline points={curve(model.sd_mean, model.sd_std)} fill="none" stroke={MODEL_COLOR} strokeWidth={2} />}
      {/* x axis */}
      <line x1={padL} y1={padT + plotH} x2={padL + plotW} y2={padT + plotH} stroke="#ccd2da" />
      {ticks.map((t, i) => (
        <text key={i} x={xPx(t)} y={H - 10} fontSize={11} fill="#445063" textAnchor="middle">{t}</text>
      ))}
    </svg>
  );
}

const NONE_HEAD = 'none' as const;
type HeadSelection = PlacementHeadKey | typeof NONE_HEAD;

// The "None" / four-head radio group that picks which placement head is
// overlaid on the board, stacked vertically below the Unseen tiles pane in
// the board's sidebar column. An option is disabled only when the whole
// `placement` payload is absent for this position -- every head has a
// Monte-Carlo truth plane whenever the payload exists, so there's always at
// least the 'sim' mode to show, even for generations with no exported ONNX.
function PlacementHeadRadios({
  placement, selected, onChange,
}: { placement: PlacementData | null | undefined; selected: HeadSelection; onChange: (h: HeadSelection) => void }) {
  const disabled = !placement;
  return (
    <div className="placement-head-picker">
      <div className="placement-head-picker-heading">Placement overlay</div>
      <span className="radio-group vertical">
        <label>
          <input type="radio" name="placement-head" checked={selected === NONE_HEAD} onChange={() => onChange(NONE_HEAD)} />
          None
        </label>
        {HEAD_OPTIONS.map((opt) => (
          <label
            key={opt.key}
            className={disabled ? 'disabled' : undefined}
            title={disabled ? 'no placement data for this position' : undefined}
          >
            <input
              type="radio"
              name="placement-head"
              checked={selected === opt.key}
              disabled={disabled}
              onChange={() => onChange(opt.key)}
            />
            {opt.label}
          </label>
        ))}
      </span>
    </div>
  );
}

const MODE_OPTIONS: { mode: OverlayMode; label: string }[] = [
  { mode: 'residual', label: 'Residual' },
  { mode: 'pred', label: 'Model' },
  { mode: 'sim', label: 'Sim' },
];

// The 3-way segmented control that picks the overlay's display mode, sitting
// directly below the head radio group. Hidden entirely when no head is
// selected (there's nothing to show a mode for). 'Residual' and 'Model' need
// an exported model prediction for the selected head, so they're disabled
// (with an explanatory tooltip) when the head's `pred` is null; 'Sim' reads
// only the Monte-Carlo truth plane and is always available.
function PlacementModeControl({
  head, mode, onChange,
}: { head: PlacementHeadData | undefined; mode: OverlayMode; onChange: (m: OverlayMode) => void }) {
  const hasPred = !!head?.pred;
  return (
    <span className="seg placement-mode-seg">
      {MODE_OPTIONS.map((opt) => {
        const disabled = opt.mode !== 'sim' && !hasPred;
        return (
          <button
            key={opt.mode}
            type="button"
            className={mode === opt.mode ? 'active' : undefined}
            disabled={disabled}
            title={disabled ? 'no exported model for this generation' : undefined}
            onClick={() => onChange(opt.mode)}
          >
            {opt.label}
          </button>
        );
      })}
    </span>
  );
}

// One color ramp within the legend: a label plus a handful of swatches at
// increasing raw values, built with the same overlayColor() scale the board
// uses, so a swatch's apparent intensity always matches a halo of the same
// value.
function LegendRamp({ hue, cap, label }: { hue: string; cap: number; label: string }) {
  return (
    <div className="placement-legend-ramp">
      <div className="placement-legend-ramp-label">{label}</div>
      <div className="placement-legend-gradient" style={{ background: overlayGradient(hue, cap) }} />
    </div>
  );
}

// The endpoint labels under a LegendRamp's gradient, shown once per legend
// rather than once per ramp: the display floor on the left edge and the
// saturation cap (everything at or beyond it renders alike) on the right.
function LegendValues({ cap }: { cap: number }) {
  return (
    <div className="placement-legend-values">
      <span>{FLOOR.toFixed(2)}</span>
      <span>{cap}+</span>
    </div>
  );
}

// The legend explaining the overlay's fixed color scale for the current
// mode, stacked below the mode control whenever an overlay is rendered.
// Every mode's ramp is built from the same fixed caps/floor the board
// overlay uses (see placementOverlay.ts), so nothing here depends on the
// position on screen.
function PlacementLegend({ mode }: { mode: OverlayMode }) {
  if (mode === 'residual') {
    return (
      <div className="legend vertical placement-legend">
        <LegendRamp hue={MODEL_HUE} cap={RESIDUAL_CAP} label="model high (pred > sim)" />
        <LegendRamp hue={SIM_HUE} cap={RESIDUAL_CAP} label="model low (pred < sim)" />
        <LegendValues cap={RESIDUAL_CAP} />
        <div className="placement-legend-floor">|residual| &lt; {FLOOR.toFixed(2)} not shown</div>
      </div>
    );
  }
  const hue = mode === 'pred' ? MODEL_HUE : SIM_HUE;
  const label = mode === 'pred' ? 'model Pr' : 'sim Pr';
  return (
    <div className="legend vertical placement-legend">
      <LegendRamp hue={hue} cap={RAW_CAP} label={label} />
      <LegendValues cap={RAW_CAP} />
      <div className="placement-legend-floor">&lt; {FLOOR.toFixed(2)} not shown</div>
    </div>
  );
}

export default function PositionEvalAnalysis({ task, tag }: { task: string; tag: string | null }) {
  const [positions, setPositions] = useState<string[]>([]);
  const [generations, setGenerations] = useState<Generation[]>([]);
  const [posIdx, setPosIdx] = useState(0);
  const [genIdx, setGenIdx] = useState(0);
  const [latest, setLatest] = useState(true);
  const [payload, setPayload] = useState<Payload | null>(null);
  const [headSel, setHeadSel] = useState<HeadSelection>(NONE_HEAD);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('residual');
  const [altOpen, setAltOpen] = useState(false);
  const [altLeave, setAltLeave] = useState('');
  const [altResult, setAltResult] = useState<AltResult | null>(null);
  const [altError, setAltError] = useState<string | null>(null);
  const [altBusy, setAltBusy] = useState(false);

  // The dataset's positions (fixed). Retry until they load: at startup the data API
  // can take ~10s to bind (it imports torch + the engine FFI).
  useEffect(() => {
    if (positions.length > 0) return;
    let cancelled = false;
    const tryFetch = () =>
      getJSON('/api/position_eval/positions')
        .then((d) => {
          if (!cancelled && d.positions.length) setPositions(d.positions);
        })
        .catch(() => {});
    tryFetch();
    const id = setInterval(tryFetch, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [positions.length]);

  useEffect(() => {
    if (!tag) return;
    const refresh = () =>
      getJSON(`/api/position_eval/generations?task=${task}&tag=${tag}`)
        .then((d: { generations: Generation[] }) =>
          // Keep the same array reference when nothing changed, so an unchanged poll
          // doesn't re-render the tab (only a genuinely new generation does).
          setGenerations((prev) => {
            const next = d.generations;
            const same =
              prev.length === next.length &&
              prev[prev.length - 1]?.generation === next[next.length - 1]?.generation;
            return same ? prev : next;
          }),
        )
        .catch(() => {});
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [task, tag]);

  const genCount = generations.length;

  useEffect(() => {
    if (latest) setGenIdx(Math.max(0, genCount - 1));
  }, [latest, genCount]);

  const effIdx = Math.min(genIdx, Math.max(0, genCount - 1));
  const effGen = generations[effIdx]?.generation ?? null;

  useEffect(() => {
    if (!tag || positions.length === 0) {
      setPayload(null);
      return;
    }
    const g = effGen == null ? 'latest' : String(effGen);
    getJSON(`/api/position_eval/position?task=${task}&tag=${tag}&position=${posIdx}&generation=${g}`)
      .then(setPayload)
      .catch(() => setPayload(null));
  }, [task, tag, positions.length, posIdx, effGen]);

  // An alternate-leave result is tied to a specific position + generation, so drop it
  // when either changes (the entered text is kept so it can be re-submitted).
  useEffect(() => {
    setAltResult(null);
    setAltError(null);
  }, [posIdx, effGen]);

  // Evaluate the selected model on the entered alternate leave. Uses a raw fetch (not
  // getJSON) so a 400's validation message reaches the UI instead of being swallowed.
  const submitAlt = async () => {
    const leave = altLeave.trim();
    if (!leave || !tag) return;
    setAltBusy(true);
    setAltError(null);
    setAltResult(null);
    try {
      const g = effGen == null ? 'latest' : String(effGen);
      const r = await fetch(
        `/api/position_eval/alt_leave?task=${task}&tag=${tag}&position=${posIdx}` +
          `&generation=${g}&leave=${encodeURIComponent(leave)}`,
      );
      const d = await r.json();
      if (!r.ok || d.error) setAltError(d.error || `request failed (${r.status})`);
      else setAltResult(d);
    } catch {
      setAltError('request failed');
    } finally {
      setAltBusy(false);
    }
  };

  const lastMoveCells = useMemo(
    () => new Set((payload?.last_move ?? []).map(([r, c]) => `${r},${c}`)),
    [payload?.last_move],
  );

  // Drop back to "None" if the newly loaded position no longer has
  // placement data at all, so the radio group never sits on a disabled
  // option.
  useEffect(() => {
    if (headSel === NONE_HEAD) return;
    if (!payload?.placement?.heads[headSel]) setHeadSel(NONE_HEAD);
  }, [payload, headSel]);

  // Fall back to 'sim' if the current mode needs a prediction the newly
  // loaded position/generation doesn't have (e.g. the generation changed to
  // one with no exported ONNX), so the mode control never sits on a
  // disabled selection.
  useEffect(() => {
    if (headSel === NONE_HEAD || overlayMode === 'sim') return;
    const head = payload?.placement?.heads[headSel];
    if (!head?.pred) setOverlayMode('sim');
  }, [payload, headSel, overlayMode]);

  const placementOverlay = useMemo(() => {
    if (headSel === NONE_HEAD || !payload) return null;
    return buildPlacementOverlay(payload.placement?.heads, headSel, overlayMode, payload.board);
  }, [headSel, payload, overlayMode]);

  if (!tag) return <div className="muted" style={{ padding: 20 }}>Select a tag.</div>;
  if (positions.length === 0) {
    return <div className="muted" style={{ padding: 20 }}>No position-evaluation dataset available.</div>;
  }

  const genLabel =
    effGen == null
      ? 'no checkpoints yet'
      : `gen ${effGen} (${generations[effIdx]?.positions.toLocaleString()} positions)`;

  return (
    <div className="lane-analysis">
      <div className="lane-controls-row">
        <span>Model</span>
        <GenerationSlider
          count={genCount}
          valueIdx={effIdx}
          follow={latest}
          onChange={(idx, f) => {
            setGenIdx(idx);
            setLatest(f);
          }}
          label={genLabel}
        />
      </div>
      <div className="lane-controls-row">
        <label>
          Position{' '}
          <button className="arrow" onClick={() => setPosIdx((i) => Math.max(0, i - 1))}>◀</button>{' '}
          <select value={posIdx} onChange={(e) => setPosIdx(Number(e.target.value))}>
            {positions.map((name, i) => (
              <option key={name} value={i}>{name}</option>
            ))}
          </select>{' '}
          <button className="arrow" onClick={() => setPosIdx((i) => Math.min(positions.length - 1, i + 1))}>▶</button>
        </label>
      </div>

      {!payload ? (
        <div className="muted" style={{ padding: 20 }}>Loading position…</div>
      ) : (
        <>
          <div style={{ fontSize: 13, color: '#2c3540', margin: '4px 2px 10px' }}>
            POV: <b>Player {payload.start_player + 1}</b> (just moved) — score{' '}
            <b>{payload.scores[0]}</b>–{payload.scores[1]}
            {!payload.has_prediction && (
              <span style={{ color: '#a05a00', marginLeft: 10 }}>· no model prediction yet</span>
            )}
          </div>

          <div className="lane-main">
            <div className="lane-board-area">
              <Board
                board={payload.board}
                bonuses={payload.bonuses}
                candidateTiles={[]}
                tileScores={payload.tile_scores}
                cursorRow={null}
                cursorCol={null}
                cursorDir={null}
                lastMoveCells={lastMoveCells}
                interactive={false}
                onCellClick={() => {}}
                onCellDrop={() => {}}
                cellHalos={placementOverlay?.halos}
              />
            </div>
            {/* The unseen pool the Monte-Carlo samples: 100 tiles minus the board and
                the POV's leave (i.e. the bag + the opponent's rack). Constrained to the
                game app's sidebar width so the tiles render at the same size. Below it,
                in the same column, sits the placement-overlay head picker, mode control,
                and legend. */}
            <div style={{ width: 320, flexShrink: 0 }}>
              <UnseenTiles
                state={{
                  type: 'state',
                  board: payload.board,
                  bonuses: payload.bonuses,
                  rack: payload.rack,
                  scores: payload.scores,
                  player_names: ['Player 1', 'Player 2'],
                  bag_count: payload.bag_count,
                  opponent_rack_count: payload.opponent_rack_count,
                  your_turn: false,
                  game_over: false,
                }}
              />
              <PlacementHeadRadios placement={payload.placement} selected={headSel} onChange={setHeadSel} />
              {headSel !== NONE_HEAD && (
                <PlacementModeControl
                  head={payload.placement?.heads[headSel]}
                  mode={overlayMode}
                  onChange={setOverlayMode}
                />
              )}
              {placementOverlay && <PlacementLegend mode={overlayMode} />}
            </div>
          </div>

          <div className="lane-rack">
            <Rack
              tiles={payload.rack}
              usedIndices={NO_USED}
              label={`Leave (Player ${payload.start_player + 1})`}
              interactive={false}
            />
            <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
              {!altOpen ? (
                <button className="arrow" style={{ padding: '4px 10px', width: 'auto' }} onClick={() => setAltOpen(true)}>
                  Alternate Leave
                </button>
              ) : (
                <>
                  <input
                    autoFocus
                    value={altLeave}
                    spellCheck={false}
                    placeholder={`${payload.rack.length} tiles, e.g. ZQU?`}
                    onChange={(e) => setAltLeave(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') submitAlt();
                      if (e.key === 'Escape') setAltOpen(false);
                    }}
                    style={{ fontFamily: 'monospace', fontSize: 14, padding: '4px 8px', width: 150, textTransform: 'uppercase' }}
                  />
                  <button className="arrow" style={{ padding: '4px 10px', width: 'auto' }} disabled={altBusy} onClick={submitAlt}>
                    {altBusy ? '…' : 'Evaluate'}
                  </button>
                </>
              )}
            </div>
            {altError && (
              <div style={{ marginTop: 6, color: '#c0392b', fontSize: 13 }}>{altError}</div>
            )}
            {altResult && <AltLeaveResult result={altResult} />}
          </div>

          <div className="lane-detail">
            <h3>Win / Loss / Draw — model vs Monte-Carlo</h3>
            <WldChart mc={payload.mc} model={payload.model} />

            <h3 style={{ marginTop: 18 }}>Final score-delta — Monte-Carlo vs model Gaussian</h3>
            <div style={{ fontSize: 12, color: '#445063', marginBottom: 4 }}>
              MC mean <b style={{ color: MC_COLOR }}>{payload.mc.score_delta_mean.toFixed(1)}</b>
              {' '}(σ {mcStd(payload.mc).toFixed(1)})
              {payload.model && (
                <>
                  {' · '}model mean <b style={{ color: MODEL_COLOR }}>{payload.model.sd_mean.toFixed(1)}</b>
                  {' '}(σ {payload.model.sd_std.toFixed(1)})
                </>
              )}
            </div>
            <ScoreDeltaChart mc={payload.mc} model={payload.model} />
          </div>
        </>
      )}
    </div>
  );
}
