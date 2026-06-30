import React from 'react';
import { DragTilePayload, PlacedTile } from '../types';

interface BoardProps {
  board: (string | null)[][];
  bonuses: (string | null)[][];
  candidateTiles: PlacedTile[];
  tileScores: Record<string, number>;
  cursorRow: number | null;
  cursorCol: number | null;
  cursorDir: 'horizontal' | 'vertical' | null;
  interactive: boolean;
  onCellClick: (row: number, col: number) => void;
  onCellDrop: (row: number, col: number, payload: DragTilePayload) => void;
  // Board squares ("row,col") of the most recent move, highlighted when set.
  lastMoveCells?: Set<string>;
  // A whole lane (a row or a column) to highlight -- used by the lane-analysis
  // view to mark the selected lane. Every cell in it gets the `lane-highlight`
  // class (tiles included), unlike the empty-cell-only `in-direction` cursor hint.
  highlightLane?: { horizontal: boolean; index: number } | null;
}

const BONUS_COLORS: Record<string, { bg: string; text: string }> = {
  DL: { bg: '#a8d8ea', text: '#1a5276' },
  TL: { bg: '#2980b9', text: '#ffffff' },
  DW: { bg: '#f5b7b1', text: '#922b21' },
  TW: { bg: '#e74c3c', text: '#ffffff' },
};

const BONUS_LABELS: Record<string, string> = {
  DL: 'DL',
  TL: 'TL',
  DW: 'DW',
  TW: 'TW',
};

const Board: React.FC<BoardProps> = ({
  board, bonuses, candidateTiles, tileScores, cursorRow, cursorCol, cursorDir,
  interactive, onCellClick, onCellDrop, lastMoveCells, highlightLane,
}) => {
  const dim = 15;

  // Build lookup map for candidate tiles.
  const candidateMap = new Map<string, PlacedTile>();
  for (const t of candidateTiles) {
    candidateMap.set(`${t.row},${t.col}`, t);
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e: React.DragEvent, row: number, col: number) => {
    e.preventDefault();
    const data = e.dataTransfer.getData('text/plain');
    try {
      const payload: DragTilePayload = JSON.parse(data);
      onCellDrop(row, col, payload);
    } catch {}
  };

  return (
    <div className="board">
      <div className="board-header">
        <div className="board-corner" />
        {Array.from({ length: dim }, (_, i) => (
          <div key={i} className="board-col-label">
            {String.fromCharCode(65 + i)}
          </div>
        ))}
      </div>
      {Array.from({ length: dim }, (_, r) => {
        const isHighlightedRow = cursorDir === 'horizontal' && cursorRow === r;
        const isHighlightedCol = cursorDir === 'vertical' && cursorCol !== null;

        return (
          <div key={r} className="board-row">
            <div className="board-row-label">{r + 1}</div>
            {Array.from({ length: dim }, (_, c) => {
              const key = `${r},${c}`;
              const tile = board[r]?.[c];
              const bonus = bonuses[r]?.[c];
              const isCenter = r === 7 && c === 7;
              const bonusStyle = bonus && BONUS_COLORS[bonus];
              const candidate = candidateMap.get(key);
              const isCursor = cursorRow === r && cursorCol === c;
              const isInDir = isHighlightedRow || (isHighlightedCol && c === cursorCol);

              let cellClass = 'board-cell';
              if (tile) cellClass += ' has-tile';
              else if (candidate) cellClass += ' has-candidate';
              else if (isCenter && !tile) cellClass += ' center';

              if (tile && lastMoveCells?.has(key)) cellClass += ' last-move';

              if (isCursor) cellClass += ' cursor-cell';
              else if (isInDir && !tile && !candidate) cellClass += ' in-direction';

              const inLane = highlightLane != null &&
                (highlightLane.horizontal ? highlightLane.index === r : highlightLane.index === c);
              if (inLane) cellClass += ' lane-highlight';

              if (interactive && !tile) cellClass += ' clickable';

              // Determine what to display in this cell.
              const isExistingTile = !!tile;
              const displayLetter = tile ?? candidate?.letter ?? null;
              // Blank detection: existing board tiles use lowercase for blanks;
              // candidate tiles have explicit isBlank flag.
              const isBlank = candidate?.isBlank ||
                (isExistingTile && tile === tile!.toLowerCase() && tile !== tile!.toUpperCase());
              // Tile score: blanks score 0; regular tiles look up from tileScores.
              const tileScore = displayLetter && !isBlank
                ? (tileScores[displayLetter.toUpperCase()] ?? null)
                : null;

              return (
                <div
                  key={c}
                  className={cellClass}
                  style={
                    !tile && !candidate && bonusStyle
                      ? { backgroundColor: bonusStyle.bg, color: bonusStyle.text }
                      : undefined
                  }
                  onClick={() => interactive && onCellClick(r, c)}
                  onDragOver={interactive && !tile ? handleDragOver : undefined}
                  onDrop={interactive && !tile ? (e) => handleDrop(e, r, c) : undefined}
                >
                  {displayLetter ? (
                    <>
                      <span className={`tile-letter${isBlank ? ' blank' : ''}`}>
                        {displayLetter.toUpperCase()}
                      </span>
                      {tileScore != null && (
                        <span className="board-tile-score">{tileScore}</span>
                      )}
                    </>
                  ) : bonus ? (
                    <span className="bonus-label">{BONUS_LABELS[bonus]}</span>
                  ) : isCenter ? (
                    <span className="center-star">★</span>
                  ) : null}
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
};

export default Board;
