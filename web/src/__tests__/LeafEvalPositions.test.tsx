import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import LeafEvalPositions from '../components/LeafEvalPositions';

// deep_bot_leaf_eval's Positions tab against a canned data API: the leaf's
// values and the matches' pick are the selected generation's, the slider
// labels a generation with its match score, a click draws a move and charts
// its value over the generations, and a move the leaf did not value says why.

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
  if (url.startsWith('/api/deep_bot/positions')) {
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
  return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({ error: 'nope' }) } as Response);
}

describe('LeafEvalPositions', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(fakeFetch));
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("opens on the newest generation's pick and ranks the moves by the leaf", async () => {
    render(<LeafEvalPositions task="deep_bot_leaf_eval" tag="v2" />);
    await waitFor(() => expect(screen.getByText('On the board: 15D .ROTIC')).toBeTruthy());
    expect(screen.getByText(/generation 20 \(8,000,000 rows\) · match 0\.545 ± 0\.012/)).toBeTruthy();
    expect(screen.getByText('leaf #53 · 20.2%')).toBeTruthy();
    expect(screen.getByText(/296 moves that keep the bag/)).toBeTruthy();
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
    await waitFor(() => expect(screen.getByText('generation 20 · 40.4%')).toBeTruthy());
    fireEvent.click(screen.getAllByText('G11 TORI')[0]);
    expect(screen.getByText('On the board: G11 TORI')).toBeTruthy();
    await waitFor(() => expect(screen.getByText('generation 20 · 20.2%')).toBeTruthy());
  });

  it('says why a move has no value', async () => {
    const unvalued = { ...payload(20), values: payload(20).values.slice(0, 1) };
    vi.stubGlobal('fetch', vi.fn((url: string) =>
      url.startsWith('/api/deep_bot_leaf_eval/position?')
        ? Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(unvalued) } as Response)
        : fakeFetch(url),
    ));
    render(<LeafEvalPositions task="deep_bot_leaf_eval" tag="v2" />);
    await waitFor(() => expect(screen.getByText('not valued (it would empty the bag)')).toBeTruthy());
  });
});
