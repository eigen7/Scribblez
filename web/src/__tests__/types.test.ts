import { describe, it, expect } from 'vitest';
import { parseMove } from '../types';

describe('parseMove', () => {
  it('parses a horizontal move starting at the centre square', () => {
    // "8H" -> row 7 (1-based "8"), col 7 ("H" = 8th column, 0-based 7).
    expect(parseMove('8H WAREZ 54')).toEqual({
      row: 7,
      col: 7,
      dir: 'horizontal',
      word: 'WAREZ',
      score: 54,
    });
  });

  it('parses a vertical move starting at the centre square', () => {
    // Vertical positions put the column letter first.
    expect(parseMove('H8 WAREZ 54')).toEqual({
      row: 7,
      col: 7,
      dir: 'vertical',
      word: 'WAREZ',
      score: 54,
    });
  });

  it('parses a vertical move with a two-digit row', () => {
    expect(parseMove('A15 ZOO 9')).toEqual({
      row: 14,
      col: 0,
      dir: 'vertical',
      word: 'ZOO',
      score: 9,
    });
  });

  it('parses a horizontal move with a two-digit row', () => {
    expect(parseMove('15O ZOO 9')).toEqual({
      row: 14,
      col: 14,
      dir: 'horizontal',
      word: 'ZOO',
      score: 9,
    });
  });

  it('preserves blank designations (lowercase) in the word field', () => {
    // Blanks render as lowercase in GCG-style move text. The parser should
    // not normalize case, since downstream callers rely on case to detect
    // blanks.
    const parsed = parseMove('8H WaREZ 54');
    expect(parsed).not.toBeNull();
    expect(parsed!.word).toBe('WaREZ');
  });

  it('returns null for pass moves', () => {
    expect(parseMove('(pass)')).toBeNull();
  });

  it('returns null for exchange moves', () => {
    expect(parseMove('(exch ABC)')).toBeNull();
  });

  it('returns null for malformed input', () => {
    expect(parseMove('hello')).toBeNull();
    expect(parseMove('8Z FOO 10')).toBeNull(); // Z is out of range
    expect(parseMove('')).toBeNull();
  });
});
