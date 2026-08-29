import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import EvidenceTrajectories from '../components/EvidenceTrajectories';

// The Trajectories tab against a canned data API: the sets / positions /
// generations lists load, the position payload renders the trajectory strip
// (cards beyond the prefix dimmed), the move table with the next sim marked,
// and clicking a card re-fetches with that slot.

const empty15 = () => Array.from({ length: 15 }, () => Array<string | null>(15).fill(null));

const sim = { n: 8, win: 0.5, draw: 0, loss: 0.5, value: 0.5, value_se: 0.17, delta_mean: 1, delta_std: 2 };
const card = (slot: number, index: number, notation: string, in_prefix: boolean, off_policy = false) => ({
  slot, index, notation, score: 10, tiles: [{ row: 7, col: 7, letter: 'A', isBlank: false }],
  lane: { horizontal: true, index: 7 }, off_policy, in_prefix, sim,
  plain_value: 0.6, cond_value: 0.55, plain_rank: index, cond_rank: index,
});
const row = (index: number, notation: string, extra: Partial<Record<string, unknown>> = {}) => ({
  index, notation, score: 10, plain_rank: index, cond_rank: index, plain_value: 0.6, cond_value: 0.55,
  gain: 0.01, slot: null, sim_value: null, next_sim: false, ...extra,
});

function payload(prefix: number, slot: number) {
  return {
    name: 'egotize-lane', set: 'face-up-trajectory-set', generation: 2,
    board: {
      board: empty15(), bonuses: empty15(), rack: [{ letter: 'A', score: 1 }], scores: [440, 387],
      player_names: ['Hasty_1', 'Hasty_2'], bag_count: 2, opponent_rack_count: 7,
      tile_scores: { A: 1 }, mover: 0, opp_leave: '', last_move: [],
    },
    prefix, max_prefix: 2, rollouts: 8, num_legal_moves: 3, trained: true, score_diff: 53,
    next_sim: 2,
    trajectory: [card(0, 0, '8H ANCHOR', prefix > 0), card(1, 1, '8H PROP', prefix > 1), card(2, 5, '8H OFFPOL', false, true)],
    moves: [row(0, '8H ANCHOR', { slot: 0, sim_value: 0.5 }), row(1, '8H PROP', { slot: 1, sim_value: 0.5 }), row(2, '8H NEXT', { next_sim: true })],
    planes: { slot, n: 8, heads: {} },
  };
}

const calls: string[] = [];

function fakeFetch(url: string): Promise<Response> {
  calls.push(url);
  const json = (body: unknown) =>
    Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) } as Response);
  if (url.startsWith('/api/evidence_trajectories/sets')) return json({ sets: ['face-up-trajectory-set'], default: 'face-up-trajectory-set' });
  if (url.startsWith('/api/evidence_trajectories/positions')) return json({ positions: ['egotize-lane'] });
  if (url.startsWith('/api/evidence_trajectories/generations')) return json({ generations: [{ generation: 0, epoch: null }, { generation: 2, epoch: 1 }] });
  if (url.startsWith('/api/evidence_trajectories/position?')) {
    const q = new URLSearchParams(url.split('?')[1]);
    const prefix = q.has('prefix') ? Number(q.get('prefix')) : 2;
    const slot = q.has('slot') ? Number(q.get('slot')) : Math.max(prefix - 1, 0);
    return json(payload(prefix, slot));
  }
  return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({ error: 'nope' }) } as Response);
}

describe('EvidenceTrajectories', () => {
  beforeEach(() => {
    calls.length = 0;
    vi.stubGlobal('fetch', vi.fn(fakeFetch));
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders the trajectory strip, dims cards beyond the prefix, and marks the next sim', async () => {
    render(<EvidenceTrajectories task="evidence_trajectories" tag="trajectories" />);
    await waitFor(() => expect(document.querySelectorAll('.traj-card').length).toBe(3));
    // Latest generation (2), largest prefix (2): anchor + proposal in the
    // prefix, the off-policy draw dimmed.
    expect(screen.getByText('gen 2 (pass 1)')).toBeInTheDocument();
    const cards = document.querySelectorAll('.traj-card');
    expect(cards[0].classList.contains('beyond')).toBe(false);
    expect(cards[1].classList.contains('beyond')).toBe(false);
    expect(cards[2].classList.contains('beyond')).toBe(true);
    expect(cards[2].textContent).toContain('off-policy');
    // The prefix's last candidate is selected by default.
    expect(cards[1].classList.contains('selected')).toBe(true);
    // The move table marks the next sim, and the header names it.
    expect(document.querySelectorAll('tr.next-sim').length).toBe(1);
    expect(screen.getByText('next sim')).toBeInTheDocument();
    expect(screen.getAllByText('8H NEXT').length).toBe(2); // the header line and the table row
  });

  it('re-fetches with the clicked card as the slot', async () => {
    render(<EvidenceTrajectories task="evidence_trajectories" tag="trajectories" />);
    await waitFor(() => expect(document.querySelectorAll('.traj-card').length).toBe(3));
    fireEvent.click(document.querySelectorAll('.traj-card')[0]);
    await waitFor(() => expect(calls.some((u) => u.includes('&slot=0'))).toBe(true));
    await waitFor(() => expect(document.querySelectorAll('.traj-card')[0].classList.contains('selected')).toBe(true));
  });
});
