import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import AppSurvey from '../AppSurvey';
import { SurveyData, SurveyMove, SurveyPosition } from '../lib/surveyTypes';

const nextMove = { score: 20, bingo_pct: 5, non_play_pct: 1, adjacent_pct: 30, score_hist: [1, 2] };

function move(name: string, rank: number, win: number, outside: boolean): SurveyMove {
  return {
    move: name,
    hasty_rank: rank,
    equity: 10,
    score: 12,
    leave: 'AB',
    is_setup: false,
    tiles: [{ row: 7, col: 7, letter: name[name.length - 1], isBlank: false }],
    stats: {
      rollouts: 5000, win_pct: win, spread: 3, spread_sd: 80, delta_hist: [0, 3, 1],
      end_swing: 2, opp_stranded: 4, self_stranded: 1, self_went_out_pct: 60,
      opp_went_out_pct: 40, opp_reply: nextMove, self_next: nextMove,
    },
    ...(outside ? { gain_pct: 6, sigmas: 4.2, beats_cut: true, versus: 'H8 TOP' } : {}),
  };
}

function position(name: string, moves: SurveyMove[]): SurveyPosition {
  return {
    name, gcg: `${name}.gcg`, turn: 12, mover: 1, rack: 'ABCDEF?', opp_known_leave: 'QU',
    opp_rack_count: 7,
    scores: [200, 180], bag_size: 9, num_legal_moves: 300, played: 'H8 TOP',
    board: Array.from({ length: 15 }, () => Array<string | null>(15).fill(null)),
    moves,
  };
}

const DATA: SurveyData = {
  survey: { cut: 10, rollouts: 1000, confirm_rollouts: 5000, recipe: 'all' },
  bonuses: Array.from({ length: 15 }, () => Array<string | null>(15).fill(null)),
  tile_scores: { A: 1, Z: 10 },
  positions: [
    position('first', [
      move('C3 FIND', 62, 55.5, true),
      move('C3 FINE', 70, 52, true),
      move('J1 MUD', 1, 48, false),
      move('H8 TOP', 2, 49.5, false),
    ]),
    position('second', [move('A1 QAX', 40, 70, true), move('B2 TOY', 1, 64, false)]),
  ],
};

const statsMoves = () =>
  [...document.querySelectorAll('.survey-stats-move')].map((el) => el.textContent);
const previewed = () => document.querySelector('.board-cell.has-candidate')?.textContent;

// Render under fake timers and let the survey fetch resolve, with the clock
// standing exactly at the moment the position mounted -- so the alternation's
// ticks land where the test advances to, however slow the machine.
async function renderOnFakeClock() {
  vi.useFakeTimers();
  render(<AppSurvey />);
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

describe('AppSurvey', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve(DATA) })));
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('opens on the best outside play beside the top move it was measured against', async () => {
    render(<AppSurvey />);
    await waitFor(() => expect(screen.getByText(/1 \/ 2 · first/)).toBeInTheDocument());
    // Both sections have a selection -- the outside play and its `versus`, not
    // hasty #1 -- and both moves' statistics show, outside first.
    expect(statsMoves()).toEqual(['C3 FIND', 'H8 TOP']);
    expect(screen.getByText('+6.0 pts, +4.2σ')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /H8 TOP/ })).toHaveAttribute('aria-pressed', 'true');
    // Each row carries the move's points.
    expect(screen.getByRole('button', { name: /C3 FIND/ })).toHaveTextContent('12');
    // The mover's view of the score, the opponent's rack (leave QU + 5 drawn), the unseen pool.
    expect(screen.getByText(/to move 180–200 \(-20\)/)).toBeInTheDocument();
    expect(document.querySelectorAll('.rack-tile.drawn-tile')).toHaveLength(5);
    expect(screen.getByText(/Unseen tiles \(14\)/)).toBeInTheDocument();
  });

  it('alternates the two selected moves on the board, each in its section tint', async () => {
    await renderOnFakeClock();
    expect(previewed()).toBe('D');
    const board = document.querySelector('.survey-board')!;
    expect(board).toHaveClass('alternating', 'showing-outside');
    // The previewed move's tiles leave the rack: FIND's D is greyed.
    expect(document.querySelectorAll('.rack-tile.used')).toHaveLength(1);
    act(() => void vi.advanceTimersByTime(3000));
    expect(previewed()).toBe('P');
    expect(board).toHaveClass('showing-hasty');
    act(() => void vi.advanceTimersByTime(3000));
    expect(previewed()).toBe('D');
  });

  it('keeps one selection per section, and a lone selection holds the board still', async () => {
    await renderOnFakeClock();
    // Another outside play replaces the outside selection; the hasty one stays.
    fireEvent.click(screen.getByRole('button', { name: /C3 FINE/ }));
    expect(statsMoves()).toEqual(['C3 FINE', 'H8 TOP']);
    // Clicking the selected hasty move deselects it: one column, no alternation.
    fireEvent.click(screen.getByRole('button', { name: /H8 TOP/ }));
    expect(statsMoves()).toEqual(['C3 FINE']);
    expect(document.querySelector('.survey-board')).not.toHaveClass('alternating');
    act(() => void vi.advanceTimersByTime(7000));
    expect(previewed()).toBe('E');
  });

  it('names each histogram bar in its own tooltip', async () => {
    render(<AppSurvey />);
    await screen.findByRole('button', { name: /C3 FIND/ });
    const titles = [...document.querySelectorAll('.survey-hist-slot')].map((el) => el.getAttribute('title'));
    // The margin histogram's first bars, then a next-move score histogram's.
    expect(titles[0]).toBe('final margin below −200: 0 rollouts (0.0%)');
    expect(titles[1]).toBe('final margin -200 to -176: 3 rollouts (75.0%)');
    expect(titles).toContain('scored 0–9: 1 rollouts (33.3%)');
    expect(titles).toContain('scored 100 or more: 2 rollouts (66.7%)');
  });

  it('steps positions with ←/→ and the active section\'s selection with ↑/↓', async () => {
    render(<AppSurvey />);
    fireEvent.click(await screen.findByRole('button', { name: /J1 MUD/ }));
    fireEvent.keyDown(window, { key: 'ArrowDown' }); // within the hasty section, just clicked
    expect(statsMoves()).toEqual(['C3 FIND', 'H8 TOP']);
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    expect(screen.getByText(/2 \/ 2 · second/)).toBeInTheDocument();
    expect(statsMoves()).toEqual(['A1 QAX']); // its `versus` is not among its moves
    expect(screen.getByRole('button', { name: 'next position' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'previous position' }));
    expect(statsMoves()).toEqual(['C3 FIND', 'H8 TOP']); // a revisited position reopens afresh
  });
});
