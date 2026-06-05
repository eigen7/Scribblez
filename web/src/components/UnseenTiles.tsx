import React from 'react';
import { GameState } from '../types';

// Standard English Scrabble tile distribution (100 tiles total). The grid
// renders one cell per tile in this order so a column always corresponds to
// the same physical tile -- as letters are played, slots empty out and the
// shape of what's unseen becomes scannable at a glance.
const DISTRIBUTION: ReadonlyArray<readonly [string, number]> = [
  ['A', 9], ['B', 2], ['C', 2], ['D', 4], ['E', 12], ['F', 2], ['G', 3], ['H', 2],
  ['I', 9], ['J', 1], ['K', 1], ['L', 4], ['M', 2], ['N', 6], ['O', 8], ['P', 2],
  ['Q', 1], ['R', 6], ['S', 4], ['T', 6], ['U', 4], ['V', 2], ['W', 2], ['X', 1],
  ['Y', 2], ['Z', 1], ['?', 2],
];

interface UnseenTilesProps {
  state: GameState;
}

// Count how many of each letter currently sit on the board. Lowercase board
// cells are designated blanks and count toward '?'.
function countBoard(board: (string | null)[][]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const row of board) {
    for (const cell of row) {
      if (!cell) continue;
      const isBlank = cell >= 'a' && cell <= 'z';
      const key = isBlank ? '?' : cell;
      out[key] = (out[key] ?? 0) + 1;
    }
  }
  return out;
}

// Count tiles in the human's own rack (the letters we already know are not
// in the bag/opponent rack).
function countRack(rack: GameState['rack']): Record<string, number> {
  const out: Record<string, number> = {};
  for (const t of rack) {
    out[t.letter] = (out[t.letter] ?? 0) + 1;
  }
  return out;
}

const UnseenTiles: React.FC<UnseenTilesProps> = ({ state }) => {
  const onBoard = countBoard(state.board);
  const inRack = countRack(state.rack);

  // Walk the distribution alphabetically; for each letter render `remaining`
  // filled slots then `total - remaining` empty slots so removed tiles vacate
  // their column from the right side of each letter's block.
  type Slot = { letter: string; present: boolean };
  const slots: Slot[] = [];
  for (const [letter, total] of DISTRIBUTION) {
    const gone = (onBoard[letter] ?? 0) + (inRack[letter] ?? 0);
    const remaining = Math.max(0, total - gone);
    for (let i = 0; i < remaining; i++) slots.push({ letter, present: true });
    for (let i = remaining; i < total; i++) slots.push({ letter, present: false });
  }
  // Sanity: the distribution sums to 100, so slots is always length 100.

  const unseenCount = state.bag_count + state.opponent_rack_count;

  return (
    <div className="unseen-tiles">
      <div className="unseen-tiles-label">Unseen tiles ({unseenCount})</div>
      <div className="unseen-tiles-grid">
        {slots.map((s, i) =>
          s.present ? (
            <div key={i} className="unseen-cell present" title={s.letter === '?' ? 'blank' : s.letter}>
              <span>{s.letter === '?' ? '' : s.letter}</span>
            </div>
          ) : (
            <div key={i} className="unseen-cell absent" />
          ),
        )}
      </div>
    </div>
  );
};

export default UnseenTiles;
