import { describe, it, expect } from 'vitest';
import {
  buildPlacementOverlay,
  overlayColor,
  FLOOR,
  RESIDUAL_CAP,
  RAW_CAP,
  MODEL_HUE,
  SIM_HUE,
  PlacementHeadData,
  PlacementHeadKey,
} from '../lib/placementOverlay';

const BOARD_DIM = 15;

const emptyBoard = (): (string | null)[][] =>
  Array.from({ length: BOARD_DIM }, () => Array.from({ length: BOARD_DIM }, () => null));

const zeros = (): number[][] =>
  Array.from({ length: BOARD_DIM }, () => Array.from({ length: BOARD_DIM }, () => 0));

const allLegal = (): boolean[][] =>
  Array.from({ length: BOARD_DIM }, () => Array.from({ length: BOARD_DIM }, () => true));

// Builds a full four-head record with every head defaulting to all-zero
// truth, an all-legal plane, and a null prediction, overriding just the head
// under test.
function headsWith(key: PlacementHeadKey, data: PlacementHeadData): Record<PlacementHeadKey, PlacementHeadData> {
  const blank = (): PlacementHeadData => ({ truth: zeros(), pred: null, legal: allLegal() });
  return {
    opp_next_placement: blank(),
    self_next_placement: blank(),
    opp_win_placement: blank(),
    self_win_placement: blank(),
    [key]: data,
  };
}

const HEAD: PlacementHeadKey = 'opp_next_placement';

describe('overlayColor', () => {
  it('reaches MAX_BLEND once v hits the cap', () => {
    const color = overlayColor(MODEL_HUE, 0.5, 0.5);
    // MODEL_HUE #3a86d4 blended toward white by MAX_BLEND = 0.95.
    expect(color).toBe('rgb(68, 140, 214)');
  });

  it('clamps values beyond the cap to the same MAX_BLEND color', () => {
    expect(overlayColor(MODEL_HUE, 5, 0.5)).toBe(overlayColor(MODEL_HUE, 0.5, 0.5));
  });
});

describe('buildPlacementOverlay', () => {
  it('returns null for residual/pred modes when the head has no prediction', () => {
    const heads = headsWith(HEAD, { truth: zeros(), pred: null, legal: allLegal() });
    expect(buildPlacementOverlay(heads, HEAD, 'residual', emptyBoard())).toBeNull();
    expect(buildPlacementOverlay(heads, HEAD, 'pred', emptyBoard())).toBeNull();
  });

  it('works in sim mode even when pred is null', () => {
    const truth = zeros();
    truth[3][3] = 0.4;
    const heads = headsWith(HEAD, { truth, pred: null, legal: allLegal() });
    const overlay = buildPlacementOverlay(heads, HEAD, 'sim', emptyBoard());
    expect(overlay).not.toBeNull();
    expect(overlay!.halos[3][3]).not.toBeNull();
    expect(overlay!.halos[3][3]!.title).toBe('sim 0.400');
  });

  it('keeps a tooltip-only halo (no ring color) when the raw value is below FLOOR', () => {
    const truth = zeros();
    truth[3][3] = FLOOR - 0.001;
    const heads = headsWith(HEAD, { truth, pred: null, legal: allLegal() });
    const overlay = buildPlacementOverlay(heads, HEAD, 'sim', emptyBoard());
    expect(overlay!.halos[3][3]).not.toBeNull();
    expect(overlay!.halos[3][3]!.color).toBeNull();
    expect(overlay!.halos[3][3]!.title).toBe(`sim ${(FLOOR - 0.001).toFixed(3)}`);
  });

  it('omits any halo entry for a masked (illegal) empty cell, even with a nonzero value', () => {
    const truth = zeros();
    truth[3][3] = 0.4; // well above FLOOR
    const legal = allLegal();
    legal[3][3] = false;
    const heads = headsWith(HEAD, { truth, pred: null, legal });
    const overlay = buildPlacementOverlay(heads, HEAD, 'sim', emptyBoard());
    expect(overlay!.halos[3][3]).toBeNull();
  });

  it('does not apply the legal check to occupied squares in sim mode', () => {
    // An occupied cell's `legal` entry is always false server-side, but sim
    // mode must keep showing it -- that's the "no rollout ever played on the
    // model's mass" mismatch this mode exists to surface.
    const truth = zeros();
    truth[3][3] = 0.4;
    const legal = allLegal();
    legal[3][3] = false;
    const heads = headsWith(HEAD, { truth, pred: null, legal });
    const board = emptyBoard();
    board[3][3] = 'A';
    const overlay = buildPlacementOverlay(heads, HEAD, 'sim', board);
    expect(overlay!.halos[3][3]).not.toBeNull();
  });

  it('draws a halo right at FLOOR', () => {
    const truth = zeros();
    truth[3][3] = FLOOR;
    const heads = headsWith(HEAD, { truth, pred: null, legal: allLegal() });
    const overlay = buildPlacementOverlay(heads, HEAD, 'sim', emptyBoard());
    expect(overlay!.halos[3][3]).not.toBeNull();
  });

  it('saturates to the cap color once the raw value reaches the cap', () => {
    const truth = zeros();
    truth[3][3] = RAW_CAP * 3; // well beyond the cap
    const heads = headsWith(HEAD, { truth, pred: null, legal: allLegal() });
    const overlay = buildPlacementOverlay(heads, HEAD, 'sim', emptyBoard());
    expect(overlay!.halos[3][3]!.color).toBe(overlayColor(SIM_HUE, RAW_CAP, RAW_CAP));
  });

  it('colors pred mode blue (MODEL_HUE)', () => {
    const truth = zeros();
    const pred = zeros();
    pred[3][3] = 0.3;
    const heads = headsWith(HEAD, { truth, pred, legal: allLegal() });
    const overlay = buildPlacementOverlay(heads, HEAD, 'pred', emptyBoard());
    expect(overlay!.halos[3][3]!.color).toBe(overlayColor(MODEL_HUE, 0.3, RAW_CAP));
  });

  it('colors sim mode red (SIM_HUE)', () => {
    const truth = zeros();
    truth[3][3] = 0.3;
    const heads = headsWith(HEAD, { truth, pred: null, legal: allLegal() });
    const overlay = buildPlacementOverlay(heads, HEAD, 'sim', emptyBoard());
    expect(overlay!.halos[3][3]!.color).toBe(overlayColor(SIM_HUE, 0.3, RAW_CAP));
  });

  it('colors residual blue when pred exceeds sim, red when it falls short', () => {
    const truth = zeros();
    truth[3][3] = 0.1;
    truth[5][5] = 0.4;
    const pred = zeros();
    pred[3][3] = 0.3; // pred > truth -> model high -> blue
    pred[5][5] = 0.1; // pred < truth -> model low -> red
    const heads = headsWith(HEAD, { truth, pred, legal: allLegal() });
    const overlay = buildPlacementOverlay(heads, HEAD, 'residual', emptyBoard());

    expect(overlay!.halos[3][3]!.color).toBe(overlayColor(MODEL_HUE, 0.2, RESIDUAL_CAP));
    expect(overlay!.halos[5][5]!.color).toBe(overlayColor(SIM_HUE, 0.3, RESIDUAL_CAP));
  });

  it('includes the full pred/sim/residual tooltip whenever pred is present', () => {
    const truth = zeros();
    truth[3][3] = 0.1;
    const pred = zeros();
    pred[3][3] = 0.3;
    const heads = headsWith(HEAD, { truth, pred, legal: allLegal() });
    const overlay = buildPlacementOverlay(heads, HEAD, 'residual', emptyBoard());
    expect(overlay!.halos[3][3]!.title).toBe('pred 0.300 / sim 0.100 / residual +0.200');
  });

  it('skips occupied squares in residual and pred modes', () => {
    const truth = zeros();
    truth[3][3] = 0.1;
    const pred = zeros();
    pred[3][3] = 0.3;
    const heads = headsWith(HEAD, { truth, pred, legal: allLegal() });
    const board = emptyBoard();
    board[3][3] = 'A';

    expect(buildPlacementOverlay(heads, HEAD, 'residual', board)!.halos[3][3]).toBeNull();
    expect(buildPlacementOverlay(heads, HEAD, 'pred', board)!.halos[3][3]).toBeNull();
  });

  it('does not skip occupied squares in sim mode', () => {
    const truth = zeros();
    truth[3][3] = 0.4;
    const heads = headsWith(HEAD, { truth, pred: null, legal: allLegal() });
    const board = emptyBoard();
    board[3][3] = 'A';

    const overlay = buildPlacementOverlay(heads, HEAD, 'sim', board);
    expect(overlay!.halos[3][3]).not.toBeNull();
  });

  it('without a legal plane (e.g. the Trajectories tab), FLOOR alone gates the halo', () => {
    const truth = zeros();
    truth[3][3] = FLOOR - 0.001;
    const heads = headsWith(HEAD, { truth, pred: null }); // no `legal` field at all
    const overlay = buildPlacementOverlay(heads, HEAD, 'sim', emptyBoard());
    expect(overlay!.halos[3][3]).toBeNull();
  });
});
