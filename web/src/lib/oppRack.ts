import { TileInfo } from '../types';

// The Positions tab's opponent rack: their last move's retained leave followed
// by the tiles drawn since. The leave is spelled out when known (face-up
// leaves; '?' in the string is a blank) and rendered as plain "?" tiles when
// hidden; the replenishment is always hidden and shaded apart (isDrawn, the
// manual tool's green) so the rack shows at a glance which tiles their last
// play accounts for.
export function oppRackTiles(
  oppLeave: string | null,
  leaveSize: number,
  rackCount: number,
  tileScores: Record<string, number>,
): TileInfo[] {
  const tiles: TileInfo[] = [];
  for (let i = 0; i < leaveSize; i++) {
    if (oppLeave == null) {
      tiles.push({ letter: '?', score: 0, isUnknown: true });
    } else if (oppLeave[i] === '?') {
      tiles.push({ letter: '', score: 0, isBlank: true });
    } else {
      tiles.push({ letter: oppLeave[i], score: tileScores[oppLeave[i]] ?? 0 });
    }
  }
  for (let i = leaveSize; i < rackCount; i++) {
    tiles.push({ letter: '?', score: 0, isUnknown: true, isDrawn: true });
  }
  return tiles;
}
