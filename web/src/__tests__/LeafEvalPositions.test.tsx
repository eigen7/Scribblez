import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import LeafEvalPositions from '../components/LeafEvalPositions';

// deep_bot_leaf_eval's Positions tab against a canned data API: the leaf's
// values and the matches' pick are the selected generation's, the slider
// labels a generation with its match score, a click draws a move and charts
// its value over the generations, a move the leaf did not value says why, and
// the opponent's rack is the tag's fixed leave inference model's.

const empty15 = () => Array.from({ length: 15 }, () => Array<null>(15).fill(null));

const move = (m: string, row = 7) => ({
  move: m, display: m, tiles: [{ row, col: 7, letter: m.slice(-1), isBlank: false }],
});

function payload(generation: number) {
  return {
    id: 'tori', title: 'TORI: a square where the Q counts twice', turn: 17, mover: 0, rack: 'CIIOQRT',
    scores: [311, 330], bag_size: 14, opp_rack_count: 7, board: empty15(), bonuses: empty15(),
    tile_scores: {}, last_move: null, reference: [move('G11 TORI', 10)],
    static_equity_best: move('15D .ROTIC', 14), generation, candidates: 10,
    pick: { ...move('15D .ROTIC', 14), reading: 'equity' },
    num_values: 296,
    values: [
      { ...move('15D .ROTIC', 14), points: 36, value: 0.404, equity_rank: 1, leaf_rank: 1 },
      { ...move('G11 TORI', 10), value: 0.202, equity_rank: 31, leaf_rank: 53 },
    ],
    match: { score: 0.545, ci_half_width: 0.012, games: 10000, opponent: '--type=hastybot-endgame' },
  };
}

function fakeFetch(url: string): Promise<Response> {
  const json = (body: unknown) =>
    Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) } as Response);
  if (url.startsWith('/api/known_positions')) {
    return json({ positions: [{ id: 'tori', title: 'TORI: a square where the Q counts twice' }] });
  }
  if (url.startsWith('/api/deep_bot_leaf_eval/generations')) {
    return json({
      generations: [
        { generation: 10, rows_trained: 4000000, score: 0.52, ci_half_width: 0.012 },
        { generation: 20, rows_trained: 8000000, score: 0.545, ci_half_width: 0.012 },
      ],
    });
  }
  if (url.startsWith('/api/deep_bot_leaf_eval/position?')) {
    const g = Number(new URLSearchParams(url.split('?')[1]).get('generation'));
    return json(payload(g));
  }
  if (url.startsWith('/api/deep_bot_leaf_eval/history?')) {
    const m = new URLSearchParams(url.split('?')[1]).get('move');
    return json({
      history: [
        { generation: 10, value: m === 'G11 TORI' ? 0.25 : 0.38 },
        { generation: 20, value: m === 'G11 TORI' ? 0.202 : 0.404 },
      ],
    });
  }
  if (url.startsWith('/api/deep_bot_leaf_eval/belief?')) {
    return json({ belief });
  }
  return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({ error: 'nope' }) } as Response);
}

const belief = {
  modelled: true, leave_length: 3, refill: 4,
  presence: [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ?'].map((tile) => ({ tile, inferred: 0.4, uniform: 0.3 })),
  leave_value: { inferred: { mean: 4, sd: 6 }, uniform: { mean: 1, sd: 8 } },
  racks: [], model: "release 0, the self-play's",
};

describe('LeafEvalPositions', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(fakeFetch));
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("opens on the newest generation's reference and ranks the moves by the leaf", async () => {
    render(<LeafEvalPositions task="deep_bot_leaf_eval" tag="v2" />);
    await waitFor(() => expect(screen.getByText('On the board: G11 TORI')).toBeTruthy());
    expect(screen.getByText(/generation 20 \(8,000,000 rows\) · match 0\.545 ± 0\.012/)).toBeTruthy();
    expect(screen.getByText('leaf #53 · 20.2%')).toBeTruthy();
    expect(screen.getByText(/296 moves, best first/)).toBeTruthy();
  });

  it("lists each move's points beside its leaf eval and equity rank", async () => {
    render(<LeafEvalPositions task="deep_bot_leaf_eval" tag="v2" />);
    await waitFor(() => expect(screen.getByText('leaf eval')).toBeTruthy());
    const headers = screen.getAllByRole('columnheader').map((h) => h.textContent);
    expect(headers).toEqual(['#', 'move', 'points', 'leaf eval', '', 'equity rank', '']);
    const row = screen.getAllByRole('row')[1];
    expect([...row.querySelectorAll('td')].map((td) => td.textContent).slice(0, 4))
      .toEqual(['1', '15D .ROTIC', '36', '40.4%']);
  });

  it('draws a clicked move on the board and charts its leaf eval', async () => {
    render(<LeafEvalPositions task="deep_bot_leaf_eval" tag="v2" />);
    await waitFor(() => expect(screen.getByText('generation 20 · 20.2%')).toBeTruthy());
    fireEvent.click(screen.getAllByText('15D .ROTIC')[0]);
    expect(screen.getByText('On the board: 15D .ROTIC')).toBeTruthy();
    await waitFor(() => expect(screen.getByText('generation 20 · 40.4%')).toBeTruthy());
  });

  it("shows the opponent's rack under the tag's fixed leave inference model", async () => {
    render(<LeafEvalPositions task="leaf_bot_generation" tag="release1" />);
    await waitFor(() => expect(screen.getByText("leave inference release 0, the self-play's")).toBeTruthy());
    expect(screen.getByText('inferred +4.0 ± 6.0 · uniform +1.0 ± 8.0')).toBeTruthy();
  });

  it('says when a move has no value', async () => {
    const unvalued = { ...payload(20), values: payload(20).values.slice(0, 1) };
    vi.stubGlobal('fetch', vi.fn((url: string) =>
      url.startsWith('/api/deep_bot_leaf_eval/position?')
        ? Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(unvalued) } as Response)
        : fakeFetch(url),
    ));
    render(<LeafEvalPositions task="deep_bot_leaf_eval" tag="v2" />);
    await waitFor(() => expect(screen.getByText('not valued')).toBeTruthy());
  });

  const withRollouts = (sameBelief: boolean) => {
    const p = payload(20);
    return {
      ...p,
      values: [
        { ...p.values[0], rollout: 0.3, rollout_rank: 2 },
        { ...p.values[1], rollout: 0.5, rollout_rank: 1 },
      ],
      rollouts: {
        tag: 't1',
        run: { leave_inference: { tag: 'li', generation: 292 }, playouts: 1024, candidates: 600 },
        same_belief: sameBelief,
        num_values: 296,
      },
    };
  };
  const serveRollouts = (sameBelief: boolean) =>
    vi.stubGlobal('fetch', vi.fn((url: string) =>
      url.startsWith('/api/deep_bot_leaf_eval/position?')
        ? Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(withRollouts(sameBelief)) } as Response)
        : fakeFetch(url),
    ));

  it("lists each move's hasty rollouts beside its leaf eval and charts them", async () => {
    serveRollouts(true);
    render(<LeafEvalPositions task="deep_bot_leaf_eval" tag="v2" />);
    await waitFor(() => expect(screen.getByText('leaf #53 · 20.2% · rollouts #1 · 50.0%')).toBeTruthy());
    const headers = screen.getAllByRole('columnheader').map((h) => h.textContent);
    expect(headers).toEqual(['#', 'move', 'points', 'leaf eval', '', 'hasty rollout win%', 'rollout rank', 'equity rank', '']);
    const row = screen.getAllByRole('row')[1];
    expect([...row.querySelectorAll('td')].map((td) => td.textContent).slice(5, 7)).toEqual(['30.0%', '#2']);
    expect(screen.getByText(/Hasty rollouts \(tag t1\)/)).toBeTruthy();
    expect(screen.queryByText(/another leave inference model/)).toBeNull();
    await waitFor(() => expect(screen.getByText('· hasty rollouts 50.0%')).toBeTruthy());
  });

  it("warns when the rollouts' opponent leaves are not drawn from the analysis's belief", async () => {
    serveRollouts(false);
    render(<LeafEvalPositions task="deep_bot_leaf_eval" tag="v2" />);
    await waitFor(() => expect(screen.getByText(/another leave inference model/)).toBeTruthy());
  });
});
