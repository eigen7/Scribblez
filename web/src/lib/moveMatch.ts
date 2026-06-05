import { GameState, MoveOption, PlacedTile, parseMove, TileInfo } from '../types';

/**
 * Find which rack tile index to consume for a given letter.
 *
 * If `preferBlank` is true, an unused blank ('?') is returned first. Otherwise
 * an exact-letter unused tile is preferred, falling back to an unused blank.
 *
 * Returns -1 if no suitable tile is available.
 */
export function findRackTile(
  rack: TileInfo[],
  usedIndices: Set<number>,
  letter: string,
  preferBlank: boolean,
): number {
  if (preferBlank) {
    const blankIdx = rack.findIndex((t, i) => !usedIndices.has(i) && t.letter === '?');
    if (blankIdx >= 0) return blankIdx;
  }
  const target = letter.toUpperCase();
  const exactIdx = rack.findIndex(
    (t, i) => !usedIndices.has(i) && t.letter.toUpperCase() === target,
  );
  if (exactIdx >= 0) return exactIdx;
  return rack.findIndex((t, i) => !usedIndices.has(i) && t.letter === '?');
}

/**
 * Given a move text, compute the new candidate tiles to display on the board
 * and the rack indices they would consume. Existing board tiles are skipped.
 *
 * Returns null if the move text doesn't parse (e.g. pass/exchange).
 */
export function candidatesFromMove(
  move: MoveOption,
  board: (string | null)[][],
  rack: TileInfo[],
): { candidates: PlacedTile[]; usedRackIndices: Set<number> } | null {
  const parsed = parseMove(move.text);
  if (!parsed) return null;

  const candidates: PlacedTile[] = [];
  const used = new Set<number>();
  for (let i = 0; i < parsed.word.length; i++) {
    const r = parsed.dir === 'vertical' ? parsed.row + i : parsed.row;
    const c = parsed.dir === 'horizontal' ? parsed.col + i : parsed.col;
    if (board[r]?.[c]) continue;
    const ch = parsed.word[i];
    const isBlank = ch === ch.toLowerCase() && ch !== ch.toUpperCase();
    const rackIdx = findRackTile(rack, used, ch, isBlank);
    if (rackIdx >= 0) used.add(rackIdx);
    candidates.push({ row: r, col: c, letter: ch.toUpperCase(), isBlank });
  }

  return { candidates, usedRackIndices: used };
}

/**
 * Find the legal move (if any) whose new-tile placements exactly match the
 * given candidate tiles. Returns the move's `index`, or -1 if no match.
 */
export function findMatchingMove(
  candidateTiles: PlacedTile[],
  moves: MoveOption[] | undefined,
  board: (string | null)[][],
): number {
  if (!moves || candidateTiles.length === 0) return -1;

  for (const move of moves) {
    const parsed = parseMove(move.text);
    if (!parsed) continue;

    // Extract newly placed positions from this legal move.
    const newTiles: { row: number; col: number; letter: string }[] = [];
    for (let i = 0; i < parsed.word.length; i++) {
      const r = parsed.dir === 'vertical' ? parsed.row + i : parsed.row;
      const c = parsed.dir === 'horizontal' ? parsed.col + i : parsed.col;
      if (!board[r]?.[c]) {
        newTiles.push({ row: r, col: c, letter: parsed.word[i] });
      }
    }

    if (newTiles.length !== candidateTiles.length) continue;

    // Every candidate must match a new tile in this move. Blank candidates
    // match lowercase letters at the same square, normal candidates match the
    // same uppercase letter.
    let allMatch = true;
    for (const candidate of candidateTiles) {
      const match = newTiles.find(
        (t) =>
          t.row === candidate.row &&
          t.col === candidate.col &&
          (candidate.isBlank
            ? t.letter === t.letter.toLowerCase() &&
              t.letter.toUpperCase() === candidate.letter.toUpperCase()
            : t.letter === candidate.letter.toUpperCase()),
      );
      if (!match) {
        allMatch = false;
        break;
      }
    }
    if (allMatch) return move.index;
  }
  return -1;
}

/**
 * Recompute the set of used rack indices from the given candidate tiles.
 * This mirrors what the UI does when tiles are removed/reordered: walk the
 * candidates in order and re-claim a rack tile for each.
 */
export function rebuildUsedIndices(
  tiles: PlacedTile[],
  rack: TileInfo[],
): Set<number> {
  const used = new Set<number>();
  for (const t of tiles) {
    const idx = findRackTile(rack, used, t.letter, t.isBlank);
    if (idx >= 0) used.add(idx);
  }
  return used;
}

// Re-export for convenience so callers can import everything from one module.
export type { GameState };
