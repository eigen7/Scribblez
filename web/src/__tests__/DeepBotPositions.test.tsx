import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import DeepBotPositions from '../components/DeepBotPositions';

// deep_bot's Positions tab against a canned data API: the policy table and the
// picks are the selected release's, a click draws a move on the board, and
// moving the slider withholds the loaded release's model outputs until the new
// release's payload lands.

const empty15 = () => Array.from({ length: 15 }, () => Array<null>(15).fill(null));
const TILES = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ?'];

const move = (m: string, row = 7) => ({
  move: m, display: m, tiles: [{ row, col: 7, letter: m.slice(-1), isBlank: false }],
});

function payload(release: number) {
  const top = release === 1 ? '8H TOP' : '8H ZERO';
  return {
    id: 'clinker', title: 'CLINKER: an S hook', turn: 14, mover: 1, rack: 'EIJLRST',
    scores: [287, 243], bag_size: 28, opp_rack_count: 7, board: empty15(), bonuses: empty15(),
    tile_scores: {}, last_move: null, reference: [move('8H .....ER', 6)],
    static_equity_best: move('E10 J..', 8), release,
    picks: { hasty: { ...move('E10 J..', 8), reading: 'equity', seconds: 0.2 } },
    belief: {
      modelled: true, leave_length: 4, refill: 3,
      presence: TILES.map((tile) => ({ tile, inferred: 0.4, uniform: 0.3 })),
      racks: [{ leave: 'AEIO', refill: 'RST' }], model: `release ${release}`,
    },
    policy: {
      num_candidates: 300, inferred: true,
      candidates: [
        { ...move(top), prob: 0.6, win_pct: 55, win_sd_pct: 9, equity_rank: 3, policy_rank: 1 },
        { ...move('E10 J..', 8), prob: 0.2, win_pct: 50, win_sd_pct: 9, equity_rank: 1, policy_rank: 2 },
        { ...move('8H .....ER', 6), prob: 0.01, win_pct: 58, win_sd_pct: 8, equity_rank: 20, policy_rank: 40 },
      ],
    },
  };
}

let releaseGen0: (() => void) | null = null;

function fakeFetch(url: string): Promise<Response> {
  const json = (body: unknown) =>
    Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) } as Response);
  if (url.startsWith('/api/deep_bot/positions')) {
    return json({ positions: [{ id: 'clinker', title: 'CLINKER: an S hook' }] });
  }
  if (url.startsWith('/api/deep_bot/releases')) {
    return json({
      releases: [0, 1].map((release) => ({
        release, rows_trained: { leave: 0, leaf: 0, policy: 1000 * release }, picked: true,
      })),
    });
  }
  if (url.startsWith('/api/deep_bot/position?')) {
    const r = new URLSearchParams(url.split('?')[1]).get('release');
    if (r === '0') return new Promise((resolve) => { releaseGen0 = () => resolve(json(payload(0)) as never); });
    return json(payload(1));
  }
  return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({ error: 'nope' }) } as Response);
}

describe('DeepBotPositions', () => {
  beforeEach(() => {
    releaseGen0 = null;
    vi.stubGlobal('fetch', vi.fn(fakeFetch));
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the newest release's policy and picks, and draws a clicked move", async () => {
    render(<DeepBotPositions task="deep_bot" tag="v1" />);
    await waitFor(() => expect(screen.getAllByText('8H TOP').length).toBeGreaterThan(0));
    expect(screen.getByText(/release 1 \(policy 1,000 rows\)/)).toBeTruthy();
    expect(screen.getByText('On the board: 8H TOP')).toBeTruthy();
    // The reference sits far down the policy, named in both lists.
    expect(screen.getByText('policy #40 · 1.0%')).toBeTruthy();
    expect(screen.getAllByText('reference')).toHaveLength(2); // its answer row and its table tag
    expect(screen.getByText('static equity, pick · hasty')).toBeTruthy();

    fireEvent.click(document.querySelector('.dbp-answer-label')!.closest('button')!);
    expect(screen.getByText('On the board: 8H .....ER')).toBeTruthy();
  });

  it('withholds the loaded release while the slider selects another', async () => {
    render(<DeepBotPositions task="deep_bot" tag="v1" />);
    await waitFor(() => expect(screen.getAllByText('8H TOP').length).toBeGreaterThan(0));

    fireEvent.change(screen.getByRole('slider'), { target: { value: '0' } });
    expect(screen.getByText(/release 0 \(policy 0 rows\)/)).toBeTruthy();
    expect(screen.getByText(/loading release/)).toBeTruthy();
    expect(screen.queryByText('Policy · release 1')).toBeNull();

    await waitFor(() => expect(releaseGen0).not.toBeNull());
    releaseGen0!();
    await waitFor(() => expect(screen.getByText('Policy · release 0')).toBeTruthy());
    expect(screen.queryByText(/loading release/)).toBeNull();
  });
});
