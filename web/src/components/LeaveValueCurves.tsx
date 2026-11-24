import { useState } from 'react';
import type { MouseEvent } from 'react';
import { LeaveValueMoments } from '../lib/surveyTypes';

// The value of the leave the opponent kept (the leave table's points), as a
// normal curve of its mean and standard deviation under the leave inference
// model beside the same under a uniform draw of the leave
// (py/scribblez/leave_inference/beliefs.py, LeaveBeliefs.leave_value). Part
// of OppRackBelief, on its surface and in its series colors; uniform is
// dashed too, so the pair does not rest on color alone. Hovering reads out
// each curve's chance that the leave is worth at most the pointer's value.

const WIDTH = 510;
const HEIGHT = 120;
const AXIS = 16; // the value ticks under the curves
const TOP = 6;
const PLOT = HEIGHT - AXIS - TOP;
const SAMPLES = 160;
// The span each side of a mean the axis covers, in standard deviations.
const SPAN_SDS = 3.5;
// The narrowest standard deviation drawn: a belief certain of one leave would
// otherwise be a spike of unbounded height.
const MIN_SD = 0.5;
const TICK_STEPS = [1, 2, 5, 10, 20, 25, 50];
const MAX_TICKS = 9;

type Series = 'inferred' | 'uniform';
const SERIES: Series[] = ['inferred', 'uniform'];

const drawnSd = (m: { sd: number }) => Math.max(m.sd, MIN_SD);

function density(x: number, m: { mean: number; sd: number }): number {
  const s = drawnSd(m);
  const z = (x - m.mean) / s;
  return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI));
}

// Abramowitz and Stegun 7.1.26, |error| < 1.5e-7.
function erf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const y = 1 - poly * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}

const cdf = (x: number, m: { mean: number; sd: number }) =>
  0.5 * (1 + erf((x - m.mean) / (drawnSd(m) * Math.SQRT2)));

interface Scale {
  lo: number;
  hi: number;
  x: (v: number) => number;
  y: (d: number) => number;
  ticks: number[];
}

function scale(v: LeaveValueMoments): Scale {
  const lo = Math.min(...SERIES.map((k) => v[k].mean - SPAN_SDS * drawnSd(v[k])));
  const hi = Math.max(...SERIES.map((k) => v[k].mean + SPAN_SDS * drawnSd(v[k])));
  const peak = Math.max(...SERIES.map((k) => density(v[k].mean, v[k])));
  const step = TICK_STEPS.find((s) => (hi - lo) / s <= MAX_TICKS) ?? TICK_STEPS[TICK_STEPS.length - 1];
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);
  return {
    lo,
    hi,
    x: (val) => ((val - lo) / (hi - lo)) * WIDTH,
    y: (d) => TOP + PLOT * (1 - d / peak),
    ticks,
  };
}

function curvePath(m: { mean: number; sd: number }, s: Scale): string {
  const pts = Array.from({ length: SAMPLES + 1 }, (_, i) => s.lo + ((s.hi - s.lo) * i) / SAMPLES);
  return pts.map((v, i) => `${i ? 'L' : 'M'}${s.x(v).toFixed(1)},${s.y(density(v, m)).toFixed(1)}`).join('');
}

const pts = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}`;
const pct = (p: number) => `${(100 * p).toFixed(0)}%`;

function readout(v: LeaveValueMoments, at: number | null): string {
  if (at == null) {
    return SERIES.map((k) => `${k} ${pts(v[k].mean)} ± ${v[k].sd.toFixed(1)}`).join(' · ');
  }
  return `worth ≤ ${pts(at)}: ${SERIES.map((k) => `${k} ${pct(cdf(at, v[k]))}`).join(', ')}`;
}

export default function LeaveValueCurves({
  value, colors,
}: {
  value: LeaveValueMoments;
  colors: Record<Series, string>;
}) {
  const [at, setAt] = useState<number | null>(null);
  const s = scale(value);
  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    setAt(s.lo + ((e.clientX - box.left) / WIDTH) * (s.hi - s.lo));
  };
  const base = s.y(0);
  const inferred = curvePath(value.inferred, s);
  return (
    <div className="leave-value">
      <div className="belief-legend">
        <span>Kept leave&apos;s value, points</span>
        <span className="belief-readout">{readout(value, at)}</span>
      </div>
      <svg className="leave-value-chart" width={WIDTH} height={HEIGHT} role="img"
        aria-label="the kept leave's value as a normal curve, inferred and uniform"
        onMouseMove={onMove} onMouseLeave={() => setAt(null)}>
        {s.ticks.map((t) => (
          <g key={t}>
            <line className={t === 0 ? 'belief-baseline' : 'belief-grid'} x1={s.x(t)} x2={s.x(t)} y1={TOP} y2={base} />
            <text className="belief-axis" x={s.x(t)} y={HEIGHT - 3} textAnchor="middle">{t}</text>
          </g>
        ))}
        <line className="belief-baseline" x1={0} x2={WIDTH} y1={base} y2={base} />
        <path d={`${inferred}L${WIDTH},${base}L0,${base}Z`} fill={colors.inferred} opacity={0.18} />
        <path className="leave-value-inferred" d={inferred} fill="none" stroke={colors.inferred} strokeWidth={2} />
        <path className="leave-value-uniform" d={curvePath(value.uniform, s)} fill="none" stroke={colors.uniform}
          strokeWidth={2} strokeDasharray="5 3" />
        {SERIES.map((k) => (
          <line key={k} x1={s.x(value[k].mean)} x2={s.x(value[k].mean)} y1={s.y(density(value[k].mean, value[k]))}
            y2={base} stroke={colors[k]} strokeWidth={1} />
        ))}
        {at != null && <line className="belief-baseline" x1={s.x(at)} x2={s.x(at)} y1={TOP} y2={base} />}
      </svg>
    </div>
  );
}
