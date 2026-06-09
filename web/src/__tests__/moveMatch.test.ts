import { describe, it, expect } from 'vitest';
import {
  findRackTile,
  findMatchingMove,
  candidatesFromMove,
  parseExchangeMove,
  rackIndicesForExchange,
  rebuildUsedIndices,
} from '../lib/moveMatch';
import { MoveOption, PlacedTile, TileInfo } from '../types';

// Helpers ---------------------------------------------------------------

const emptyBoard = (): (string | null)[][] =>
  Array.from({ length: 15 }, () => Array.from({ length: 15 }, () => null));

const rack = (letters: string): TileInfo[] =>
  letters.split('').map((l) => ({ letter: l, score: l === '?' ? 0 : 1 }));

// Tests -----------------------------------------------------------------

describe('findRackTile', () => {
  it('returns the first unused exact-letter tile', () => {
    const r = rack('ABCDE');
    expect(findRackTile(r, new Set(), 'C', false)).toBe(2);
  });

  it('respects usedIndices', () => {
    const r = rack('AABCD');
    // Index 0 already consumed, must pick index 1.
    expect(findRackTile(r, new Set([0]), 'A', false)).toBe(1);
  });

  it('falls back to a blank when no exact match exists', () => {
    const r = rack('AB?DE');
    expect(findRackTile(r, new Set(), 'Z', false)).toBe(2);
  });

  it('returns -1 when neither exact match nor blank exists', () => {
    const r = rack('ABCDE');
    expect(findRackTile(r, new Set(), 'Z', false)).toBe(-1);
  });

  it('prefers a blank over an exact match when preferBlank is true', () => {
    const r = rack('A?');
    // Letter 'A' exists at 0, but preferBlank should give us the blank at 1.
    expect(findRackTile(r, new Set(), 'A', true)).toBe(1);
  });

  it('falls back to exact when preferBlank is requested but no blank is free', () => {
    const r = rack('A?');
    // Blank is consumed; preferBlank should not prevent us from using exact.
    expect(findRackTile(r, new Set([1]), 'A', true)).toBe(0);
  });
});

describe('rebuildUsedIndices', () => {
  it('reclaims rack tiles in order, preferring exact matches', () => {
    const r = rack('TILES?');
    const tiles: PlacedTile[] = [
      { row: 7, col: 7, letter: 'T', isBlank: false },
      { row: 7, col: 8, letter: 'I', isBlank: false },
      { row: 7, col: 9, letter: 'X', isBlank: true }, // uses the blank
    ];
    const used = rebuildUsedIndices(tiles, r);
    expect(used).toEqual(new Set([0, 1, 5]));
  });

  it('returns an empty set for no tiles', () => {
    expect(rebuildUsedIndices([], rack('ABC'))).toEqual(new Set());
  });
});

describe('candidatesFromMove', () => {
  it('builds candidates for a horizontal move on an empty board', () => {
    const move: MoveOption = { index: 0, text: '8H WORD 20', score: 20 };
    const result = candidatesFromMove(move, emptyBoard(), rack('WORDABC'));
    expect(result).not.toBeNull();
    expect(result!.candidates).toEqual([
      { row: 7, col: 7, letter: 'W', isBlank: false },
      { row: 7, col: 8, letter: 'O', isBlank: false },
      { row: 7, col: 9, letter: 'R', isBlank: false },
      { row: 7, col: 10, letter: 'D', isBlank: false },
    ]);
    expect(result!.usedRackIndices).toEqual(new Set([0, 1, 2, 3]));
  });

  it('skips squares already occupied on the board (hook plays)', () => {
    const board = emptyBoard();
    board[7][7] = 'W'; // existing W means we only need ORD as new tiles.
    const move: MoveOption = { index: 0, text: '8H WORD 20', score: 20 };
    const result = candidatesFromMove(move, board, rack('ORDABC'));
    expect(result).not.toBeNull();
    expect(result!.candidates).toEqual([
      { row: 7, col: 8, letter: 'O', isBlank: false },
      { row: 7, col: 9, letter: 'R', isBlank: false },
      { row: 7, col: 10, letter: 'D', isBlank: false },
    ]);
  });

  it('marks lowercase letters in move text as blanks', () => {
    const move: MoveOption = { index: 0, text: '8H WoRD 20', score: 20 };
    const result = candidatesFromMove(move, emptyBoard(), rack('WRD?'));
    expect(result).not.toBeNull();
    // The blank should render as uppercase 'O' but isBlank=true.
    expect(result!.candidates[1]).toEqual({
      row: 7,
      col: 8,
      letter: 'O',
      isBlank: true,
    });
  });

  it('handles vertical moves by walking down the column', () => {
    const move: MoveOption = { index: 0, text: 'H8 WORD 20', score: 20 };
    const result = candidatesFromMove(move, emptyBoard(), rack('WORD'));
    expect(result).not.toBeNull();
    expect(result!.candidates.map((c) => [c.row, c.col])).toEqual([
      [7, 7],
      [8, 7],
      [9, 7],
      [10, 7],
    ]);
  });

  it('returns null for unparseable move text (pass/exchange)', () => {
    expect(
      candidatesFromMove(
        { index: 0, text: '(pass)', score: 0 },
        emptyBoard(),
        rack('ABC'),
      ),
    ).toBeNull();
  });
});

describe('findMatchingMove', () => {
  const board = emptyBoard();
  const moves: MoveOption[] = [
    { index: 0, text: '8H WORD 20', score: 20 },
    { index: 1, text: 'H8 WORD 20', score: 20 },
    { index: 2, text: '(pass)', score: 0 },
  ];

  it('returns -1 when there are no candidate tiles', () => {
    expect(findMatchingMove([], moves, board)).toBe(-1);
  });

  it('returns -1 when no move matches the candidates', () => {
    const candidates: PlacedTile[] = [
      { row: 0, col: 0, letter: 'Q', isBlank: false },
    ];
    expect(findMatchingMove(candidates, moves, board)).toBe(-1);
  });

  it('matches a horizontal move regardless of candidate insertion order', () => {
    const candidates: PlacedTile[] = [
      // Intentionally out of left-to-right order.
      { row: 7, col: 10, letter: 'D', isBlank: false },
      { row: 7, col: 7, letter: 'W', isBlank: false },
      { row: 7, col: 8, letter: 'O', isBlank: false },
      { row: 7, col: 9, letter: 'R', isBlank: false },
    ];
    expect(findMatchingMove(candidates, moves, board)).toBe(0);
  });

  it('matches a vertical move when squares line up in a column', () => {
    const candidates: PlacedTile[] = [
      { row: 7, col: 7, letter: 'W', isBlank: false },
      { row: 8, col: 7, letter: 'O', isBlank: false },
      { row: 9, col: 7, letter: 'R', isBlank: false },
      { row: 10, col: 7, letter: 'D', isBlank: false },
    ];
    expect(findMatchingMove(candidates, moves, board)).toBe(1);
  });

  it('requires blank candidates to match lowercase letters in the move text', () => {
    const blankMoves: MoveOption[] = [
      // "WoRD" — the O is a blank.
      { index: 7, text: '8H WoRD 20', score: 20 },
      // Plain WORD (no blanks) at the same location.
      { index: 8, text: '8H WORD 20', score: 20 },
    ];
    const blankCandidates: PlacedTile[] = [
      { row: 7, col: 7, letter: 'W', isBlank: false },
      { row: 7, col: 8, letter: 'O', isBlank: true },
      { row: 7, col: 9, letter: 'R', isBlank: false },
      { row: 7, col: 10, letter: 'D', isBlank: false },
    ];
    expect(findMatchingMove(blankCandidates, blankMoves, board)).toBe(7);
  });

  it('does not match a non-blank candidate against a blank in the move', () => {
    const blankMoves: MoveOption[] = [
      { index: 7, text: '8H WoRD 20', score: 20 },
    ];
    const candidates: PlacedTile[] = [
      { row: 7, col: 7, letter: 'W', isBlank: false },
      { row: 7, col: 8, letter: 'O', isBlank: false }, // not a blank
      { row: 7, col: 9, letter: 'R', isBlank: false },
      { row: 7, col: 10, letter: 'D', isBlank: false },
    ];
    expect(findMatchingMove(candidates, blankMoves, board)).toBe(-1);
  });

  it('ignores already-placed board tiles when comparing new-tile counts', () => {
    const hookBoard = emptyBoard();
    hookBoard[7][7] = 'W'; // existing tile; move only places ORD.
    const hookMoves: MoveOption[] = [
      { index: 5, text: '8H WORD 20', score: 20 },
    ];
    const candidates: PlacedTile[] = [
      { row: 7, col: 8, letter: 'O', isBlank: false },
      { row: 7, col: 9, letter: 'R', isBlank: false },
      { row: 7, col: 10, letter: 'D', isBlank: false },
    ];
    expect(findMatchingMove(candidates, hookMoves, hookBoard)).toBe(5);
  });

  it('returns -1 when moves is undefined', () => {
    const candidates: PlacedTile[] = [
      { row: 7, col: 7, letter: 'A', isBlank: false },
    ];
    expect(findMatchingMove(candidates, undefined, board)).toBe(-1);
  });
});

describe('parseExchangeMove', () => {
  it('parses an exchange move into its letters', () => {
    expect(parseExchangeMove('exch AQWW')).toEqual(['A', 'Q', 'W', 'W']);
  });

  it('treats a blank marker as its own letter', () => {
    expect(parseExchangeMove('exch A?')).toEqual(['A', '?']);
  });

  it('returns null for plays and passes', () => {
    expect(parseExchangeMove('8H WAREZ 54')).toBeNull();
    expect(parseExchangeMove('pass')).toBeNull();
  });
});

describe('rackIndicesForExchange', () => {
  it('claims one rack tile per requested letter', () => {
    const r = rack('AQWWXYZ');
    expect(rackIndicesForExchange(r, ['A', 'Q', 'W', 'W'])).toEqual(new Set([0, 1, 2, 3]));
  });

  it('does not reuse a tile for a duplicated letter beyond supply', () => {
    const r = rack('AWXYZBC');
    // Only one W available; the second W request finds nothing more to claim.
    expect(rackIndicesForExchange(r, ['W', 'W'])).toEqual(new Set([1]));
  });

  it('claims a blank for a "?" request', () => {
    const r = rack('AB?DEFG');
    expect(rackIndicesForExchange(r, ['?'])).toEqual(new Set([2]));
  });
});
