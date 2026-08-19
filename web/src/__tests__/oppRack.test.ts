import { describe, it, expect } from 'vitest';
import { oppRackTiles } from '../lib/oppRack';

const SCORES = { A: 1, C: 3, E: 1, I: 1, N: 1, S: 1 };

describe('oppRackTiles', () => {
  it('spells out a face-up leave and shades the drawn tiles', () => {
    expect(oppRackTiles('ACE', 3, 7, SCORES)).toEqual([
      { letter: 'A', score: 1 },
      { letter: 'C', score: 3 },
      { letter: 'E', score: 1 },
      { letter: '?', score: 0, isUnknown: true, isDrawn: true },
      { letter: '?', score: 0, isUnknown: true, isDrawn: true },
      { letter: '?', score: 0, isUnknown: true, isDrawn: true },
      { letter: '?', score: 0, isUnknown: true, isDrawn: true },
    ]);
  });

  it('renders a face-up blank as a blank tile, not an unknown', () => {
    expect(oppRackTiles('A?', 2, 3, SCORES)).toEqual([
      { letter: 'A', score: 1 },
      { letter: '', score: 0, isBlank: true },
      { letter: '?', score: 0, isUnknown: true, isDrawn: true },
    ]);
  });

  it('renders a hidden leave as unshaded unknowns, apart from the shaded draws', () => {
    expect(oppRackTiles(null, 2, 4, SCORES)).toEqual([
      { letter: '?', score: 0, isUnknown: true },
      { letter: '?', score: 0, isUnknown: true },
      { letter: '?', score: 0, isUnknown: true, isDrawn: true },
      { letter: '?', score: 0, isUnknown: true, isDrawn: true },
    ]);
  });

  it('an empty leave (the opponent bingoed) is all drawn tiles', () => {
    expect(oppRackTiles('', 0, 2, SCORES)).toEqual([
      { letter: '?', score: 0, isUnknown: true, isDrawn: true },
      { letter: '?', score: 0, isUnknown: true, isDrawn: true },
    ]);
  });

  it('a short rack near the end of the bag pads nothing', () => {
    expect(oppRackTiles('IN', 2, 2, SCORES)).toEqual([
      { letter: 'I', score: 1 },
      { letter: 'N', score: 1 },
    ]);
  });
});
