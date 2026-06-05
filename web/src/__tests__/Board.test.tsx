import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Board from '../components/Board';
import { PlacedTile } from '../types';

// Helpers ---------------------------------------------------------------

const emptyBoard = (): (string | null)[][] =>
  Array.from({ length: 15 }, () => Array.from({ length: 15 }, () => null));

const emptyBonuses = emptyBoard;

const TILE_SCORES: Record<string, number> = {
  A: 1, B: 3, C: 3, D: 2, E: 1, F: 4, G: 2, H: 4, I: 1, J: 8,
  K: 5, L: 1, M: 3, N: 1, O: 1, P: 3, Q: 10, R: 1, S: 1, T: 1,
  U: 1, V: 4, W: 4, X: 8, Y: 4, Z: 10,
};

interface RenderOptions {
  board?: (string | null)[][];
  bonuses?: (string | null)[][];
  candidateTiles?: PlacedTile[];
  cursorRow?: number | null;
  cursorCol?: number | null;
  cursorDir?: 'horizontal' | 'vertical' | null;
  interactive?: boolean;
  onCellClick?: (row: number, col: number) => void;
  onCellDrop?: (row: number, col: number, letter: string, isBlank: boolean) => void;
}

function renderBoard(opts: RenderOptions = {}) {
  const props = {
    board: opts.board ?? emptyBoard(),
    bonuses: opts.bonuses ?? emptyBonuses(),
    candidateTiles: opts.candidateTiles ?? [],
    tileScores: TILE_SCORES,
    cursorRow: opts.cursorRow ?? null,
    cursorCol: opts.cursorCol ?? null,
    cursorDir: opts.cursorDir ?? null,
    interactive: opts.interactive ?? true,
    onCellClick: opts.onCellClick ?? vi.fn(),
    onCellDrop: opts.onCellDrop ?? vi.fn(),
  };
  return { ...render(<Board {...props} />), props };
}

// Looks up a cell by its 1-based row/letter address (e.g. (8, 'H') = centre).
function cellAt(container: HTMLElement, row: number, colLetter: string): HTMLElement {
  // The Board renders rows as .board-row containing cells; the first child of
  // each row is the .board-row-label, followed by 15 .board-cell elements.
  const rows = container.querySelectorAll('.board-row');
  const rowEl = rows[row - 1] as HTMLElement | undefined;
  if (!rowEl) throw new Error(`row ${row} not found`);
  const cells = rowEl.querySelectorAll('.board-cell');
  const colIdx = colLetter.charCodeAt(0) - 65;
  const cell = cells[colIdx] as HTMLElement | undefined;
  if (!cell) throw new Error(`col ${colLetter} not found`);
  return cell;
}

// Tests -----------------------------------------------------------------

describe('Board', () => {
  it('renders 15 row labels and 15 column labels', () => {
    const { container } = renderBoard();
    const colLabels = container.querySelectorAll('.board-col-label');
    expect(colLabels).toHaveLength(15);
    expect(colLabels[0]).toHaveTextContent('A');
    expect(colLabels[14]).toHaveTextContent('O');

    const rowLabels = container.querySelectorAll('.board-row-label');
    expect(rowLabels).toHaveLength(15);
    expect(rowLabels[0]).toHaveTextContent('1');
    expect(rowLabels[14]).toHaveTextContent('15');
  });

  it('renders a star on the centre square when it is empty', () => {
    const { container } = renderBoard();
    const centre = cellAt(container, 8, 'H');
    expect(centre).toHaveTextContent('★');
    expect(centre).toHaveClass('center');
  });

  it('renders existing board tiles with letter and score', () => {
    const board = emptyBoard();
    board[7][7] = 'W'; // 8H
    const { container } = renderBoard({ board });
    const centre = cellAt(container, 8, 'H');
    expect(centre).toHaveClass('has-tile');
    expect(centre).toHaveTextContent('W');
    expect(centre).toHaveTextContent('4'); // W is worth 4
  });

  it('renders blank tiles (lowercase on the board) without a score', () => {
    const board = emptyBoard();
    board[7][7] = 'w'; // blank designated as W
    const { container } = renderBoard({ board });
    const centre = cellAt(container, 8, 'H');
    expect(centre).toHaveTextContent('W');
    // No tile-score for blanks.
    expect(centre.querySelector('.board-tile-score')).toBeNull();
  });

  it('renders candidate tiles with the has-candidate class', () => {
    const candidateTiles: PlacedTile[] = [
      { row: 7, col: 7, letter: 'Q', isBlank: false },
    ];
    const { container } = renderBoard({ candidateTiles });
    const centre = cellAt(container, 8, 'H');
    expect(centre).toHaveClass('has-candidate');
    expect(centre).toHaveTextContent('Q');
    expect(centre).toHaveTextContent('10');
  });

  it('marks the cursor cell and the horizontal direction line', () => {
    const { container } = renderBoard({
      cursorRow: 7,
      cursorCol: 7,
      cursorDir: 'horizontal',
    });
    expect(cellAt(container, 8, 'H')).toHaveClass('cursor-cell');
    // Same row, different column: should be highlighted as in-direction.
    expect(cellAt(container, 8, 'A')).toHaveClass('in-direction');
    expect(cellAt(container, 8, 'O')).toHaveClass('in-direction');
    // Different row, same column: NOT highlighted in horizontal mode.
    expect(cellAt(container, 1, 'H')).not.toHaveClass('in-direction');
  });

  it('marks the cursor cell and the vertical direction line', () => {
    const { container } = renderBoard({
      cursorRow: 7,
      cursorCol: 7,
      cursorDir: 'vertical',
    });
    expect(cellAt(container, 8, 'H')).toHaveClass('cursor-cell');
    // Same column, different row: in-direction.
    expect(cellAt(container, 1, 'H')).toHaveClass('in-direction');
    expect(cellAt(container, 15, 'H')).toHaveClass('in-direction');
    // Same row, different column: NOT highlighted.
    expect(cellAt(container, 8, 'A')).not.toHaveClass('in-direction');
  });

  it('marks empty cells as clickable only when interactive', () => {
    const { container: c1 } = renderBoard({ interactive: true });
    expect(cellAt(c1, 1, 'A')).toHaveClass('clickable');

    const { container: c2 } = renderBoard({ interactive: false });
    expect(cellAt(c2, 1, 'A')).not.toHaveClass('clickable');
  });

  it('does not mark occupied cells as clickable', () => {
    const board = emptyBoard();
    board[0][0] = 'A';
    const { container } = renderBoard({ board, interactive: true });
    expect(cellAt(container, 1, 'A')).not.toHaveClass('clickable');
  });

  it('invokes onCellClick with the clicked row/col when interactive', async () => {
    const onCellClick = vi.fn();
    const { container } = renderBoard({ onCellClick });
    await userEvent.click(cellAt(container, 8, 'H'));
    expect(onCellClick).toHaveBeenCalledTimes(1);
    expect(onCellClick).toHaveBeenCalledWith(7, 7);
  });

  it('does not invoke onCellClick when not interactive', async () => {
    const onCellClick = vi.fn();
    const { container } = renderBoard({ onCellClick, interactive: false });
    await userEvent.click(cellAt(container, 8, 'H'));
    expect(onCellClick).not.toHaveBeenCalled();
  });

  it('shows bonus labels on empty bonus squares', () => {
    const bonuses = emptyBoard();
    bonuses[7][7] = 'DW';
    const { container } = renderBoard({ bonuses });
    expect(cellAt(container, 8, 'H')).toHaveTextContent('DW');
  });

  it('hides bonus labels when a candidate covers the bonus square', () => {
    const bonuses = emptyBoard();
    bonuses[7][7] = 'DW';
    const candidateTiles: PlacedTile[] = [
      { row: 7, col: 7, letter: 'A', isBlank: false },
    ];
    const { container } = renderBoard({ bonuses, candidateTiles });
    const centre = cellAt(container, 8, 'H');
    expect(centre).not.toHaveTextContent('DW');
    expect(centre).toHaveTextContent('A');
  });
});

describe('Board accessibility / sanity', () => {
  it('exposes the centre star via getByText for quick smoke tests', () => {
    renderBoard();
    expect(screen.getByText('★')).toBeInTheDocument();
  });
});
