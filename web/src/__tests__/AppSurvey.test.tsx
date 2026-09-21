import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
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
    position('first', [move('C3 FIND', 62, 55.5, true), move('H8 TOP', 1, 49.5, false)]),
    position('second', [move('A1 QAX', 40, 70, true), move('B2 TOY', 1, 64, false)]),
  ],
};

describe('AppSurvey', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve(DATA) })));
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('opens on the first position with its best outside play selected', async () => {
    render(<AppSurvey />);
    await waitFor(() => expect(screen.getByText(/1 \/ 2 · first/)).toBeInTheDocument());
    // The outside play leads the list, above the top moves, and fills the stats panel.
    expect(screen.getByText('Best-simming plays outside the top')).toBeInTheDocument();
    expect(document.querySelector('.survey-stats-move')).toHaveTextContent('C3 FIND');
    expect(screen.getByText('+6.0 win pts, +4.2σ')).toBeInTheDocument();
    // Its tile is previewed on the board; the mover's view of the score is shown.
    expect(document.querySelector('.board-cell.has-candidate')).toHaveTextContent('D');
    // ...and marked used on the rack, which so shows the leave.
    expect(document.querySelectorAll('.rack-tile.used')).toHaveLength(1);
    expect(screen.getByText(/to move 180–200 \(-20\)/)).toBeInTheDocument();
  });

  it('shows the clicked move on the board and in the stats panel', async () => {
    render(<AppSurvey />);
    fireEvent.click(await screen.findByRole('button', { name: /H8 TOP/ }));
    expect(document.querySelector('.survey-stats-move')).toHaveTextContent('H8 TOP');
    expect(document.querySelector('.board-cell.has-candidate')).toHaveTextContent('P');
    expect(screen.queryByText(/win pts/)).not.toBeInTheDocument();
  });

  it('steps between positions with the arrow keys and buttons, resetting the selection', async () => {
    render(<AppSurvey />);
    fireEvent.click(await screen.findByRole('button', { name: /H8 TOP/ }));
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    expect(screen.getByText(/2 \/ 2 · second/)).toBeInTheDocument();
    expect(document.querySelector('.survey-stats-move')).toHaveTextContent('A1 QAX');
    expect(screen.getByRole('button', { name: 'next position' })).toBeDisabled();
    fireEvent.keyDown(window, { key: 'ArrowDown' });
    expect(document.querySelector('.survey-stats-move')).toHaveTextContent('B2 TOY');
    fireEvent.click(screen.getByRole('button', { name: 'previous position' }));
    expect(screen.getByText(/1 \/ 2 · first/)).toBeInTheDocument();
  });
});
