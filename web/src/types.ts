export interface TileInfo {
  letter: string;
  score: number;
  isBlank?: boolean;
  isUnknown?: boolean;
}

export interface DragTilePayload {
  letter: string;
  score?: number;
  isBlank: boolean;
  rackIndex?: number;
  fromBag?: boolean;
  player?: number;
}

export interface MoveOption {
  index: number;
  text: string;
  score: number;
  // Macondo's static equity for this play, or null if Macondo isn't
  // available (no binary configured / subprocess failure) or didn't return
  // this particular play. The UI renders null cells blank.
  equity?: number | null;
}

export interface GameState {
  type: 'state' | 'game_over';
  board: (string | null)[][];
  bonuses: (string | null)[][];
  rack: TileInfo[];
  scores: [number, number];
  player_names: [string, string];
  bag_count: number;
  opponent_rack_count: number;
  your_turn: boolean;
  game_over: boolean;
  moves?: MoveOption[];
  winner?: number;
  final_scores?: [number, number];
  tile_scores?: Record<string, number>; // e.g. { A: 1, B: 3, ... Z: 10 }
  // [row, col] board squares of the move that produced this position. Present
  // in offline position dumps (for highlighting); absent in live play.
  last_move?: [number, number][];
}

// A tile placed on the board by the user (candidate, not yet submitted).
export interface PlacedTile {
  row: number;
  col: number;
  letter: string; // uppercase letter
  isBlank: boolean; // true if using a blank tile
}

// Direction for keyboard tile entry.
export type EntryDirection = 'horizontal' | 'vertical';

// Parsed move location from move text like "8H WAREZ 54".
export interface ParsedMove {
  row: number; // 0-based
  col: number; // 0-based
  dir: 'horizontal' | 'vertical';
  word: string; // full word including existing tiles
  score: number;
}

// Parse a move text string into structured data.
// Horizontal: "8H WAREZ 54" (row=7, col=7, dir=horizontal)
// Vertical: "H8 WAREZ 54" (col=7, row=7, dir=vertical)
// Exchange: "(exch ABC)" or Pass: "(pass)" → returns null
export function parseMove(text: string): ParsedMove | null {
  if (text.startsWith('(')) return null;
  const parts = text.split(' ');
  if (parts.length < 3) return null;

  const pos = parts[0];
  const word = parts[1];
  const score = parseInt(parts[2], 10);

  // If first char is a digit, it's horizontal: row then col
  if (/^\d/.test(pos)) {
    const match = pos.match(/^(\d+)([A-O])$/);
    if (!match) return null;
    return { row: parseInt(match[1], 10) - 1, col: match[2].charCodeAt(0) - 65, dir: 'horizontal', word, score };
  } else {
    const match = pos.match(/^([A-O])(\d+)$/);
    if (!match) return null;
    return { row: parseInt(match[2], 10) - 1, col: match[1].charCodeAt(0) - 65, dir: 'vertical', word, score };
  }
}
