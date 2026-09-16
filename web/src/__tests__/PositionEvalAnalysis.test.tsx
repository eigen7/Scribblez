import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import PositionEvalAnalysis from '../components/PositionEvalAnalysis';

// The Positions tab against a canned data API: the model outputs shown are
// the selected generation's. Moving the slider withholds the loaded
// generation's bars (a spinner stands in) until the new generation's payload
// lands, so a stale prediction is never read under the new label.

const empty15 = () => Array.from({ length: 15 }, () => Array<null>(15).fill(null));

// Model win probability per generation: what the bars must show for each.
const MODEL_WIN: Record<number, number> = { 0: 0.3, 1: 0.7 };

function payload(generation: number) {
  const win = MODEL_WIN[generation];
  return {
    name: 'pos-01', start_player: 0, last_move: [], board: empty15(), bonuses: empty15(),
    rack: [], tile_scores: {}, scores: [300, 280], bag_count: 20, opponent_rack_count: 7,
    face_up_leaves: true, opp_leave: '', opp_leave_size: 0, generation, has_prediction: true,
    mc: { n: 100, wld: { win: 0.9, loss: 0.1, draw: 0 }, score_delta_hist: [[-5, 50], [5, 50]], score_delta_mean: 0 },
    model: { wld: { win, loss: 1 - win, draw: 0 }, sd_mean: 10, sd_std: 5 },
    placement: null,
  };
}

// The position fetch for generation 0 is held until the test releases it, so
// the in-flight state can be observed.
let releaseGen0: (() => void) | null = null;

function fakeFetch(url: string): Promise<Response> {
  const json = (body: unknown) =>
    Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) } as Response);
  if (url.startsWith('/api/position_eval/positions')) return json({ positions: ['pos-01'] });
  if (url.startsWith('/api/position_eval/generations')) {
    return json({ generations: [{ generation: 0, positions: 1000 }, { generation: 1, positions: 2000 }] });
  }
  if (url.startsWith('/api/position_eval/position?')) {
    const g = new URLSearchParams(url.split('?')[1]).get('generation');
    if (g === '0') return new Promise((resolve) => { releaseGen0 = () => resolve(json(payload(0)) as never); });
    return json(payload(1));
  }
  return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({ error: 'nope' }) } as Response);
}

describe('PositionEvalAnalysis', () => {
  beforeEach(() => {
    releaseGen0 = null;
    vi.stubGlobal('fetch', vi.fn(fakeFetch));
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('withholds the loaded model outputs while the slider selects another generation', async () => {
    render(<PositionEvalAnalysis task="position_eval" tag="t" />);
    // Latest (generation 1) loads with its model bars.
    await waitFor(() => expect(screen.getByText('70%')).toBeTruthy());
    expect(screen.getByText(/gen 1 \(2,000 positions\)/)).toBeTruthy();

    // Slide to generation 0: the label follows at once, the gen-1 bars go, a
    // spinner stands in, and nothing has landed yet.
    fireEvent.change(screen.getByRole('slider'), { target: { value: '0' } });
    expect(screen.getByText(/gen 0 \(1,000 positions\)/)).toBeTruthy();
    expect(screen.queryByText('70%')).toBeNull();
    expect(screen.getByText(/evaluating gen 0/)).toBeTruthy();
    expect(document.querySelector('.scz-spinner')).toBeTruthy();
    // The Monte-Carlo bars are the position's, not the generation's, and stay.
    expect(screen.getByText('90%')).toBeTruthy();

    // The debounced fetch goes out; releasing it shows generation 0's bars.
    await waitFor(() => expect(releaseGen0).not.toBeNull());
    releaseGen0!();
    await waitFor(() => expect(screen.getByText('30%')).toBeTruthy());
    expect(screen.queryByText(/evaluating gen/)).toBeNull();
    expect(document.querySelector('.scz-spinner')).toBeNull();
  });
});
