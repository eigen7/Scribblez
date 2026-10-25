import { useCallback, useEffect, useState } from 'react';
import { RackBeliefChart } from '../lib/surveyTypes';

// The blind-spots viewer's opponent-rack panel: what the mover believes the
// opponent holds, from the leave inference model (py/scribblez/blind_spots_viewer.py,
// BeliefPanel). A grouped bar per tile -- the chance the rack holds one under
// the model, beside the same under no information -- and sample racks, the
// kept leave apart from the refill, redrawn on demand.

// Series colors, validated (dataviz validate_palette.js, dark mode) against
// the panel's own surface, --belief-surface in index.css.
const INFERRED = '#3987e5';
const UNIFORM = '#c98500';

const WIDTH = 510;
const HEIGHT = 140;
const AXIS = 16; // the tile letters under the bars
const TOP = 6;
const BAR = 6;
const GAP = 2; // between a tile's two bars
const RADIUS = 2;

interface Rack {
  leave: string;
  refill: string;
}

const pct = (p: number) => `${(100 * p).toFixed(1)}%`;

// A bar rounded at its data end, square on the baseline.
function barPath(x: number, y: number, h: number): string {
  if (h <= 0) return '';
  const r = Math.min(RADIUS, h, BAR / 2);
  const base = y + h;
  return `M${x},${base} V${y + r} Q${x},${y} ${x + r},${y} H${x + BAR - r} Q${x + BAR},${y} ${x + BAR},${y + r} V${base} Z`;
}

function Chart({ chart, onHover }: { chart: RackBeliefChart; onHover: (i: number | null) => void }) {
  const group = WIDTH / chart.presence.length;
  const plot = HEIGHT - AXIS - TOP;
  const y = (p: number) => TOP + plot * (1 - p);
  return (
    <svg
      className="belief-chart"
      width={WIDTH}
      height={HEIGHT}
      role="img"
      aria-label="the chance the opponent's rack holds each tile"
      onMouseLeave={() => onHover(null)}
    >
      {[0.5, 1].map((p) => (
        <line key={p} className="belief-grid" x1={0} x2={WIDTH} y1={y(p)} y2={y(p)} />
      ))}
      <line className="belief-baseline" x1={0} x2={WIDTH} y1={y(0)} y2={y(0)} />
      {chart.presence.map((t, i) => {
        const left = i * group + (group - (2 * BAR + GAP)) / 2;
        return (
          <g key={t.tile} onMouseEnter={() => onHover(i)}>
            {/* The hit target: the whole column, bigger than either bar. */}
            <rect x={i * group} y={0} width={group} height={HEIGHT} fill="transparent" />
            <path d={barPath(left, y(t.inferred), plot * t.inferred)} fill={INFERRED} />
            <path d={barPath(left + BAR + GAP, y(t.uniform), plot * t.uniform)} fill={UNIFORM} />
            <text className="belief-axis" x={i * group + group / 2} y={HEIGHT - 3} textAnchor="middle">
              {t.tile}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function SampleRack({ rack }: { rack: Rack }) {
  return (
    <div className="belief-sample">
      {[...rack.leave].map((ch, i) => (
        <span key={`l${i}`} className="belief-tile kept">{ch}</span>
      ))}
      {[...rack.refill].map((ch, i) => (
        <span key={`r${i}`} className="belief-tile drawn">{ch}</span>
      ))}
    </div>
  );
}

// Sample racks from /api/leave_samples, a fresh seed per draw.
function useSamples(position: number) {
  const [racks, setRacks] = useState<Rack[] | null>(null);
  const [error, setError] = useState('');
  const draw = useCallback(() => {
    const seed = Math.floor(Math.random() * 2 ** 31);
    fetch(`/api/leave_samples?position=${position}&seed=${seed}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: { racks: Rack[] }) => setRacks(d.racks))
      .catch((e) => setError(String(e)));
  }, [position]);
  useEffect(draw, [draw]);
  return { racks, error, draw };
}

export default function OppRackBelief({ chart, position }: { chart: RackBeliefChart; position: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const { racks, error, draw } = useSamples(position);
  const shown = hover === null ? null : chart.presence[hover];
  return (
    <div className="belief-panel">
      <div className="survey-stat-title">Opponent&apos;s rack</div>
      <div className="belief-sub">
        {chart.modelled
          ? `kept ${chart.leave_length} at their last move, drew ${chart.refill} · P(holds at least one)`
          : `no leave inference here, so inferred is uniform: a uniform rack of ${chart.refill} · P(holds at least one)`}
      </div>
      <div className="belief-legend">
        <span>
          <i style={{ background: INFERRED }} /> inferred
        </span>
        <span>
          <i style={{ background: UNIFORM }} /> uniform
        </span>
        <span className="belief-readout">
          {shown && `${shown.tile}: inferred ${pct(shown.inferred)}, uniform ${pct(shown.uniform)}`}
        </span>
      </div>
      <Chart chart={chart} onHover={setHover} />
      <div className="belief-samples-head">
        <span>
          Sample racks: <span className="belief-tile kept">kept</span>{' '}
          <span className="belief-tile drawn">drawn</span>
        </span>
        <button type="button" onClick={draw}>
          New sample
        </button>
      </div>
      {error && <div className="belief-sub">Could not sample: {error}</div>}
      <div className="belief-samples">{racks?.map((r) => <SampleRack key={r.leave + r.refill} rack={r} />)}</div>
    </div>
  );
}
