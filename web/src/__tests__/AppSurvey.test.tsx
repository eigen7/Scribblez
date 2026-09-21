import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import AppSurvey from '../AppSurvey';
import { SurveyData, SurveyMove, SurveyPosition } from '../lib/surveyTypes';

const nextMove = {
  score: 20, bingo_pct: 5, bingo_spot: { at: 'E12', pct: 3.5 }, adjacent_pct: 30, score_hist: [1, 2],
};

// A one-tile move laying its last letter at row 7 of `col`.
function move(name: string, rank: number, win: number, outside: boolean, col = 7): SurveyMove {
  return {
    move: name,
    display: name.replace('FIND', 'F(I)ND'),
    hasty_rank: rank,
    equity: 10,
    score: 12,
    leave: 'AB',
    is_setup: false,
    tiles: [{ row: 7, col, letter: name[name.length - 1], isBlank: false }],
    stats: {
      rollouts: 5000, win_pct: win, spread: 3, spread_sd: 80, delta_hist: [0, 3, 1],
      end_swing: 2, opp_stranded: 4, self_stranded: 1, self_passed_pct: 7.5,
      opp_passed_pct: 0, self_went_out_pct: 60,
      opp_went_out_pct: 40, opp_reply: nextMove, self_next: nextMove,
    },
    ...(outside ? { gain_pct: 6, sigmas: 4.2, beats_cut: true, versus: 'H8 TOP' } : {}),
  };
}

function position(name: string, moves: SurveyMove[]): SurveyPosition {
  return {
    name, gcg: `${name}.gcg`, turn: 12, mover: 1, rack: 'ABCDEF?', opp_known_leave: 'QU',
    opp_rack_count: 7, solved_endgames: true,
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
      move('J1 MUD', 1, 48, false, 3), // clear of the others' square
      move('H8 TOP', 2, 49.5, false),
    ]),
    position('second', [move('A1 QAX', 40, 70, true), move('B2 TOY', 1, 64, false)]),
  ],
};

const statsMoves = () =>
  [...document.querySelectorAll('.survey-stats-move')].map((el) => el.textContent);
const previewed = () =>
  [...document.querySelectorAll('.board-cell.has-candidate .tile-letter')].map((el) => el.textContent);
const tints = () =>
  [...document.querySelectorAll<HTMLElement>('.board-cell-highlight')].map(
    (el) => el.style.backgroundColor,
  );

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

  it('opens on the best outside play beside hasty #1, both on the board', async () => {
    render(<AppSurvey />);
    await waitFor(() => expect(screen.getByText(/1 \/ 2 · first/)).toBeInTheDocument());
    expect(statsMoves()).toEqual(['C3 F(I)ND', 'J1 MUD']);
    expect(screen.getByText('+6.0 pts, +4.2σ')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /J1 MUD/ })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: /C3 F\(I\)ND/ })).toHaveTextContent('12'); // its points
    // The two moves share no square, so both sit on the board, each in its
    // section's tint, with no alternation; the rack stays whole.
    expect(previewed()).toEqual(['D', 'D']); // MUD's D at col 3, FIND's at col 7
    expect(new Set(tints()).size).toBe(2);
    expect(document.querySelector('.survey-board')).not.toHaveClass('alternating');
    expect(document.querySelectorAll('.rack-tile.used')).toHaveLength(0);
    // Each rack is labelled with its player's score; the mover is seat 1.
    expect(screen.getByText('To move · 180 points')).toBeInTheDocument();
    expect(screen.getByText(/Opponent · 200 points/)).toBeInTheDocument();
    expect(document.querySelectorAll('.rack-tile.drawn-tile')).toHaveLength(5);
    expect(screen.getByText(/Unseen tiles \(14\)/)).toBeInTheDocument();
  });

  it('lines the two stats panels up row for row', async () => {
    render(<AppSurvey />);
    await screen.findByRole('button', { name: /C3 F\(I\)ND/ });
    const [outside, hasty] = [...document.querySelectorAll('.survey-stats')];
    const labels = (panel: Element) =>
      [...panel.querySelectorAll('.survey-stat-row, .survey-stat-title, .survey-hist')].map(
        (el) => el.className,
      );
    expect(labels(hasty)).toEqual(labels(outside));
    expect(screen.queryByText('high-value setup')).not.toBeInTheDocument();
    expect(screen.queryByText('exchange / pass')).not.toBeInTheDocument();
    // The commonest bingo spot under each bingo row; passes in the game-end block.
    expect(screen.getAllByText('bingo@E12')).toHaveLength(4);
    expect(screen.getAllByText('we pass')).toHaveLength(2);
    expect(screen.getAllByText('7.5%')).toHaveLength(2);
    expect(screen.getByText(/endgames solved/)).toBeInTheDocument();
  });

  it('alternates two selected moves that share a square', async () => {
    await renderOnFakeClock();
    fireEvent.click(screen.getByRole('button', { name: /H8 TOP/ })); // same square as FIND
    expect(document.querySelector('.survey-board')).toHaveClass('alternating');
    expect(previewed()).toEqual(['D']);
    expect(tints()).toEqual(['rgba(230, 126, 34, 0.55)']);
    // The one move on the board leaves the rack: FIND's D is greyed.
    expect(document.querySelectorAll('.rack-tile.used')).toHaveLength(1);
    act(() => void vi.advanceTimersByTime(3000));
    expect(previewed()).toEqual(['P']);
    expect(tints()).toEqual(['rgba(142, 110, 220, 0.6)']);
    act(() => void vi.advanceTimersByTime(3000));
    expect(previewed()).toEqual(['D']);
  });

  it('keeps one selection per section, and a lone selection holds the board still', async () => {
    await renderOnFakeClock();
    // Another outside play replaces the outside selection; the hasty one stays.
    fireEvent.click(screen.getByRole('button', { name: /C3 FINE/ }));
    expect(statsMoves()).toEqual(['C3 FINE', 'J1 MUD']);
    // Clicking the selected hasty move deselects it: one column, one move.
    fireEvent.click(screen.getByRole('button', { name: /J1 MUD/ }));
    expect(statsMoves()).toEqual(['C3 FINE']);
    act(() => void vi.advanceTimersByTime(7000));
    expect(previewed()).toEqual(['E']);
  });

  it('names each histogram bar in its own tooltip, on a scale the panels share', async () => {
    render(<AppSurvey />);
    await screen.findByRole('button', { name: /C3 F\(I\)ND/ });
    const slots = [...document.querySelectorAll<HTMLElement>('.survey-hist-slot')];
    const titles = slots.map((el) => el.getAttribute('title'));
    // The margin histogram's first bars, then a next-move score histogram's.
    expect(titles[0]).toBe('final margin below −200: 0 rollouts (0.0%)');
    expect(titles[1]).toBe('final margin -200 to -176: 3 rollouts (75.0%)');
    expect(titles).toContain('scored 100 or more: 2 rollouts (66.7%)');
    // The tallest bin across both panels spans the full width.
    expect(slots[1].querySelector<HTMLElement>('.survey-hist-bar')!.style.width).toBe('100%');
  });

  it('steps positions with ←/→ and the active section\'s selection with ↑/↓', async () => {
    render(<AppSurvey />);
    fireEvent.click(await screen.findByRole('button', { name: /H8 TOP/ }));
    fireEvent.keyDown(window, { key: 'ArrowUp' }); // within the hasty section, just clicked
    expect(statsMoves()).toEqual(['C3 F(I)ND', 'J1 MUD']);
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    expect(screen.getByText(/2 \/ 2 · second/)).toBeInTheDocument();
    expect(statsMoves()).toEqual(['A1 QAX', 'B2 TOY']); // hasty #1 again, whatever was selected before
    expect(screen.getByRole('button', { name: 'next position' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'previous position' }));
    expect(statsMoves()).toEqual(['C3 F(I)ND', 'J1 MUD']); // a revisited position reopens afresh
  });
});
