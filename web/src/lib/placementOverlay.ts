// Board-overlay math for the Positions tab's placement-head heat map.
//
// Each of the model's four placement heads predicts, per board square, the
// probability of some event -- "the opponent's next move covers this square"
// for the Positions tab's collapsed view, "is anchored at this square" for the
// Trajectories pane's marginals. Monte-Carlo rollouts give an empirical
// ground-truth probability for the same event. This module turns one head's
// (truth, pred) pair, plus
// a display mode, into a per-cell "halo" -- a ring color plus a hover
// tooltip -- that a caller hands to <Board cellHalos=... />.
//
// Every mode uses a fixed, absolute color scale: a cell's color depth (an
// opaque white-to-hue blend) is a function of its raw value and a fixed cap,
// with no per-position or per-generation normalization. The same value
// always renders as the same color everywhere, so a viewer can compare
// intensity across positions and generations by eye, and the sidebar
// legend's ramps (built with the same overlayColor() helper) always match
// what's drawn on the board.

export type PlacementHeadKey =
  | 'opp_next_placement'
  | 'self_next_placement'
  | 'opp_win_placement'
  | 'self_win_placement';

export interface PlacementHeadData {
  truth: number[][];
  pred: number[][] | null;
}

export interface PlacementData {
  n: number;
  heads: Record<PlacementHeadKey, PlacementHeadData>;
}

export interface HeadOption {
  key: PlacementHeadKey;
  label: string;
}

// The five radio options offered in the UI, in display order ("None" is
// handled separately by the caller since it isn't a real head key).
export const HEAD_OPTIONS: HeadOption[] = [
  { key: 'opp_next_placement', label: 'Pr[opp play]' },
  { key: 'self_next_placement', label: 'Pr[self play]' },
  { key: 'opp_win_placement', label: 'Pr[opp play & win]' },
  { key: 'self_win_placement', label: 'Pr[self play & win]' },
];

// The board overlay has three display modes, picked in the UI by a
// segmented control next to the head radio group:
//   - 'residual': |pred - sim|, signed blue/red by which side is higher.
//     Requires an exported model prediction for the selected head.
//   - 'pred': the model's raw predicted probability (blue). Requires a
//     prediction.
//   - 'sim': the raw Monte-Carlo probability (red). Always available, since
//     it comes from the rollouts rather than the model.
export type OverlayMode = 'residual' | 'pred' | 'sim';

// The overlay's two hues: blue for the model's prediction (used for 'pred'
// and for the "model high" side of 'residual'), red for the Monte-Carlo
// ground truth (used for 'sim' and for the "model low" side of 'residual').
export const MODEL_HUE = '#3a86d4';
export const SIM_HUE = '#e74c3c';

const BOARD_DIM = 15;

// A cell's raw value must reach this floor before it gets a halo at all, in
// every mode -- below it the ring is omitted rather than drawn faint, so the
// overlay doesn't turn the whole board into a haze of near-invisible rings.
export const FLOOR = 0.02;

// The value at which a 'residual' cell reaches full intensity (MAX_BLEND).
export const RESIDUAL_CAP = 0.3;

// The value at which a 'pred' or 'sim' cell reaches full intensity.
export const RAW_CAP = 0.5;

// Exponent applied to a cell's value/cap ratio before scaling to the blend
// fraction, so intensity rises steeply off the floor and flattens out near
// the cap (values well below the cap still read as clearly visible, not
// washed out).
const GAMMA = 0.7;

// The blend-fraction range the scale maps onto. MIN_BLEND lifts the faintest
// drawn ring well off white so the hue is readable at the display floor -- a
// +0.02 and a -0.02 residual must read as blue vs pink, not as two
// near-identical off-whites. MAX_BLEND is reached once a cell's value hits
// its mode's cap (0.95 leaves even a saturated ring a hair off the pure hue).
export const MIN_BLEND = 0.25;
export const MAX_BLEND = 0.95;

export interface CellHalo {
  color: string;
  title: string;
}

export interface PlacementOverlay {
  halos: (CellHalo | null)[][];
}

// Mixes `hex` toward white: t = 1 is the pure hue, t = 0 is white. The
// result is fully opaque -- intensity is encoded by paleness, not by
// transparency, so a ring's rendered color is identical on every square
// (bonus squares included) instead of blending with the square's own
// background color.
function towardWhite(hex: string, t: number): string {
  const n = parseInt(hex.slice(1), 16);
  const mix = (ch: number) => Math.round(255 + (ch - 255) * t);
  return `rgb(${mix((n >> 16) & 255)}, ${mix((n >> 8) & 255)}, ${mix(n & 255)})`;
}

// Maps a raw value to a color on the fixed scale for one hue, used by both
// the board overlay (below) and the sidebar legend's swatches so the two
// always agree. `v` is clamped to `cap` first, so values at or beyond the
// cap all render at the same maximum intensity.
export function overlayColor(hue: string, v: number, cap: number): string {
  const t = MIN_BLEND + (MAX_BLEND - MIN_BLEND) * Math.pow(Math.min(v, cap) / cap, GAMMA);
  return towardWhite(hue, t);
}

// A CSS linear-gradient running the fixed scale for one hue from FLOOR (the
// smallest value drawn on the board) up to `cap`. The color curve is
// nonlinear in the value (the GAMMA exponent), so the gradient is sampled
// from overlayColor() at several evenly spaced stops rather than
// interpolating the two endpoints.
export function overlayGradient(hue: string, cap: number): string {
  const stops: string[] = [];
  const n = 8;
  for (let i = 0; i <= n; i++) {
    const v = FLOOR + (i / n) * (cap - FLOOR);
    stops.push(`${overlayColor(hue, v, cap)} ${((100 * i) / n).toFixed(1)}%`);
  }
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

// Builds the per-cell halo overlay for one placement head in one display
// mode. Returns null when the head has no data, or when 'residual'/'pred'
// is requested but the head has no exported model prediction to show or
// compare against ('sim' has no such requirement, since it only reads the
// Monte-Carlo truth plane). `board` is the current position's board (the
// same array passed to <Board board=... />): in 'residual' and 'pred' modes
// a square already occupied by a tile is skipped even though the head's
// prediction plane is unmasked and may be nonzero there, since a placement
// probability on an occupied square isn't a meaningful prediction. 'sim'
// mode has no such gate -- a square with model mass that no rollout ever
// played on is exactly the kind of mismatch this mode is for showing.
export function buildPlacementOverlay(
  heads: Record<PlacementHeadKey, PlacementHeadData> | undefined | null,
  key: PlacementHeadKey,
  mode: OverlayMode,
  board: (string | null)[][],
): PlacementOverlay | null {
  if (!heads) return null;
  const head = heads[key];
  if (mode !== 'sim' && !head.pred) return null;
  const { truth, pred } = head;

  const halos: (CellHalo | null)[][] = Array.from({ length: BOARD_DIM }, () =>
    new Array(BOARD_DIM).fill(null),
  );
  for (let r = 0; r < BOARD_DIM; r++) {
    for (let c = 0; c < BOARD_DIM; c++) {
      if (mode !== 'sim' && board[r][c] != null) continue;

      const truthV = truth[r][c];
      const predV = pred ? pred[r][c] : null;
      const residual = predV != null ? predV - truthV : null;

      let v: number;
      let hue: string;
      let cap: number;
      if (mode === 'residual') {
        v = Math.abs(residual!);
        hue = residual! >= 0 ? MODEL_HUE : SIM_HUE;
        cap = RESIDUAL_CAP;
      } else if (mode === 'pred') {
        v = predV!;
        hue = MODEL_HUE;
        cap = RAW_CAP;
      } else {
        v = truthV;
        hue = SIM_HUE;
        cap = RAW_CAP;
      }
      if (v < FLOOR) continue;

      const title =
        residual != null
          ? `pred ${predV!.toFixed(3)} / sim ${truthV.toFixed(3)} / residual ${residual >= 0 ? '+' : ''}${residual.toFixed(3)}`
          : `sim ${truthV.toFixed(3)}`;

      halos[r][c] = { color: overlayColor(hue, v, cap), title };
    }
  }
  return { halos };
}
