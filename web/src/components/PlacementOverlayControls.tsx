// The Positions tab's placement-overlay sidebar controls, shared with the
// evidence_trajectories Trajectories tab: the head radio group, the
// residual | model | sim segmented control, and the fixed-scale legend. The
// overlay math itself lives in lib/placementOverlay.ts.
import {
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

export const NONE_HEAD = 'none' as const;
export type HeadSelection = PlacementHeadKey | typeof NONE_HEAD;

// The "None" / four-head radio group that picks which placement head is
// overlaid on the board, stacked vertically in the board's sidebar column.
// An option is disabled only when the whole
// `placement` payload is absent for this position -- every head has a
// Monte-Carlo truth plane whenever the payload exists, so there's always at
// least the 'sim' mode to show, even for generations with no exported ONNX.
export function PlacementHeadRadios({
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
export function PlacementModeControl({
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
export function PlacementLegend({ mode }: { mode: OverlayMode }) {
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
