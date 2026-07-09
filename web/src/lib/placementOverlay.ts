// Board-overlay math for the Positions tab's placement-head residual heat map.
//
// Each of the model's four placement heads predicts, per board square, the
// probability of some event (e.g. "the opponent's next move covers this
// square"). Monte-Carlo rollouts give an empirical ground-truth probability
// for the same event. This module turns one head's (truth, pred) pair into
// a per-cell "halo" -- a ring color plus a hover tooltip -- that a caller
// hands to <Board cellHalos=... />.

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

// Each head's seat has one "marginal" head -- opp_next_placement for the
// opponent's two heads, self_next_placement for the POV player's two heads
// -- whose truth plane records every square that seat played on in at least
// one Monte-Carlo rollout. A head's halo is only drawn on squares its seat's
// marginal truth is positive, so a win head (whose own truth is 0 on squares
// that were played but never in a winning rollout) still shows those squares
// rather than hiding them.
const MARGINAL_HEAD: Record<PlacementHeadKey, PlacementHeadKey> = {
  opp_next_placement: 'opp_next_placement',
  opp_win_placement: 'opp_next_placement',
  self_next_placement: 'self_next_placement',
  self_win_placement: 'self_next_placement',
};

// Diverging residual colors, reusing the tab's existing model/Monte-Carlo
// hues so the overlay speaks the same color language as the rest of the
// view: blue marks a square where the model's prediction exceeds the
// Monte-Carlo estimate ("model high"), red marks one where it falls short
// ("model low").
export const OVERESTIMATE_COLOR = '#3a86d4';
export const UNDERESTIMATE_COLOR = '#e74c3c';

const BOARD_DIM = 15;

// A ring's minimum and maximum opacity. Every gated square gets a visible
// ring (so the overlaid area reads as a whole), with intensity growing from
// that floor up to near-opaque as |residual| approaches the board's max for
// the selected head.
const MIN_ALPHA = 0.1;
const MAX_ALPHA = 0.9;

export interface CellHalo {
  color: string;
  title: string;
}

export interface PlacementOverlay {
  halos: (CellHalo | null)[][];
  maxAbsResidual: number;
}

function withAlpha(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
}

// Builds the per-cell halo overlay for one placement head. A square is
// included only when it's gated in -- its seat's marginal head (see
// MARGINAL_HEAD) has positive truth there, meaning that seat played on it in
// at least one Monte-Carlo rollout. Squares occupied by the current position
// are never played on in a rollout, so their marginal truth is always 0 and
// they're excluded by the same gate. For each included square the residual
// is pred - truth, colored on the diverging blue/red scale above and
// normalized against the largest |residual| among the included squares (the
// ones actually shown), so the overlay always uses its full color range for
// the position on screen. Returns null when the head has no exported
// prediction to compare against.
export function buildPlacementOverlay(
  heads: Record<PlacementHeadKey, PlacementHeadData> | undefined | null,
  key: PlacementHeadKey,
): PlacementOverlay | null {
  if (!heads) return null;
  const head = heads[key];
  if (!head.pred) return null;
  const { truth, pred } = head;
  const gate = heads[MARGINAL_HEAD[key]].truth;

  let maxAbsResidual = 0;
  for (let r = 0; r < BOARD_DIM; r++) {
    for (let c = 0; c < BOARD_DIM; c++) {
      if (gate[r][c] <= 0) continue;
      maxAbsResidual = Math.max(maxAbsResidual, Math.abs(pred[r][c] - truth[r][c]));
    }
  }
  const scale = maxAbsResidual > 0 ? maxAbsResidual : 1;

  const halos: (CellHalo | null)[][] = Array.from({ length: BOARD_DIM }, () =>
    new Array(BOARD_DIM).fill(null),
  );
  for (let r = 0; r < BOARD_DIM; r++) {
    for (let c = 0; c < BOARD_DIM; c++) {
      if (gate[r][c] <= 0) continue;
      const truthV = truth[r][c];
      const predV = pred[r][c];
      const residual = predV - truthV;
      const frac = Math.min(1, Math.abs(residual) / scale);
      const alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * frac;
      const color = residual >= 0 ? OVERESTIMATE_COLOR : UNDERESTIMATE_COLOR;
      const sign = residual >= 0 ? '+' : '';
      halos[r][c] = {
        color: withAlpha(color, alpha),
        title: `pred ${predV.toFixed(3)} / sim ${truthV.toFixed(3)} / residual ${sign}${residual.toFixed(3)}`,
      };
    }
  }
  return { halos, maxAbsResidual };
}
