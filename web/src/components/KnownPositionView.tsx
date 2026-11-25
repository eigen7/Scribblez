import { ReactNode, useEffect, useMemo, useState } from 'react';
import Board from './Board';
import Rack from './Rack';
import UnseenTiles from './UnseenTiles';
import { PlacedTile, TileInfo } from '../types';
import { getJSON } from '../lib/api';

// What the leaf Positions tab (deep_bot_leaf_eval's and leaf_bot_generation's)
// shows of a known position (positions/NWL23/known-positions): the board before the move with
// a move previewed on it, both racks, the unseen tiles, and the picker over
// the positions. A tab adds its models' outputs beside them.

const NOOP = () => {};
const DIM = 15;
// How a pick reads against the position's answer (deep_bot/known_positions.py):
// the reference move, static equity's move, any other.
export const READING_COLOR: Record<string, string> = {
  reference: '#2ca02c',
  equity: '#888888',
  other: '#e67e22',
};
const PREVIEW_TINT = 'rgba(58, 134, 212, 0.55)';

export interface MoveRef {
  move: string; // GCG notation, the key the lists match on
  display: string;
  tiles: PlacedTile[];
}

// A known position as deep_bot/position_view.py's board_view gives it, with
// the board's drawing data.
export interface BoardView {
  id: string;
  title: string;
  turn: number;
  mover: number;
  rack: string;
  scores: [number, number];
  bag_size: number;
  opp_rack_count: number;
  board: (string | null)[][];
  bonuses: (string | null)[][];
  tile_scores: Record<string, number>;
  last_move: { text: string; squares: [number, number][] } | null;
  reference: MoveRef[];
  static_equity_best: MoveRef;
}

export interface KnownPosition {
  id: string;
  title: string;
}

function rackTiles(rack: string, tileScores: Record<string, number>): TileInfo[] {
  return [...rack].map((ch) =>
    ch === '?' ? { letter: '', score: 0, isBlank: true } : { letter: ch, score: tileScores[ch] ?? 0 },
  );
}

// The rack slots a move's tiles come from (a blank's slot is '?'), so the rack
// shows the leave of the move on the board.
function usedRackIndices(rack: string, move: MoveRef | null): Set<number> {
  const used = new Set<number>();
  for (const t of move?.tiles ?? []) {
    const want = t.isBlank ? '?' : t.letter;
    const slot = [...rack].findIndex((ch, i) => ch === want && !used.has(i));
    if (slot >= 0) used.add(slot);
  }
  return used;
}

function previewTints(move: MoveRef | null) {
  const grid: ({ color: string } | null)[][] = Array.from({ length: DIM }, () =>
    Array.from({ length: DIM }, () => null),
  );
  for (const t of move?.tiles ?? []) grid[t.row][t.col] = { color: PREVIEW_TINT };
  return grid;
}

// A probability as a percentage: one decimal, or two significant figures
// below 1%, so a long tail still reads apart from zero.
export const pct = (p: number) => `${p >= 0.01 ? (100 * p).toFixed(1) : (100 * p).toPrecision(2)}%`;

export function BoardColumn({ p, preview }: { p: BoardView; preview: MoveRef | null }) {
  const lastMoveCells = useMemo(
    () => new Set((p.last_move?.squares ?? []).map(([r, c]) => `${r},${c}`)),
    [p.last_move],
  );
  return (
    <div className="dbp-board">
      <Rack
        tiles={Array.from({ length: p.opp_rack_count }, () => ({ letter: '?', score: 0, isUnknown: true }))}
        usedIndices={new Set()}
        label={`Opponent (Player ${2 - p.mover}) · ${p.scores[1 - p.mover]} points`}
        interactive={false}
        hideScoreForQuestion
      />
      <Board
        board={p.board}
        bonuses={p.bonuses}
        candidateTiles={preview?.tiles ?? []}
        tileScores={p.tile_scores}
        cursorRow={null}
        cursorCol={null}
        cursorDir={null}
        interactive={false}
        onCellClick={NOOP}
        onCellDrop={NOOP}
        cellHighlights={previewTints(preview)}
        lastMoveCells={lastMoveCells}
      />
      <div className="dbp-note">
        {preview ? `On the board: ${preview.display}` : 'Click a move to draw it on the board.'}
        {p.last_move && ` · their last move: ${p.last_move.text}`}
      </div>
      <Rack
        tiles={rackTiles(p.rack, p.tile_scores)}
        usedIndices={usedRackIndices(p.rack, preview)}
        label={`To move (Player ${p.mover + 1}) · ${p.scores[p.mover]} points`}
        interactive={false}
        onTileClick={NOOP}
      />
    </div>
  );
}

export function Unseen({ p }: { p: BoardView }) {
  return (
    <UnseenTiles
      alsoSeen=""
      state={{
        type: 'state',
        board: p.board,
        bonuses: p.bonuses,
        rack: rackTiles(p.rack, p.tile_scores).map((t) => ({ ...t, letter: t.isBlank ? '?' : t.letter })),
        scores: p.scores,
        player_names: ['Player 1', 'Player 2'],
        bag_count: p.bag_size,
        opponent_rack_count: p.opp_rack_count,
        your_turn: false,
        game_over: false,
      }}
    />
  );
}

// One move of the position's answers panel: what it is (`label`), the move,
// and what the tab's model makes of it (`note`). Clicking draws it.
export function AnswerRow({
  label, move, color, note, selected, onSelect,
}: {
  label: string;
  move: MoveRef;
  color?: string;
  note: string;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button className={`dbp-answer${selected ? ' selected' : ''}`} onClick={onSelect}>
      <span className="dbp-answer-label">{label}</span>
      <span className="dbp-answer-move" style={color ? { color } : undefined}>{move.display}</span>
      <span className="dbp-answer-policy">{note}</span>
    </button>
  );
}

// The position's title line: which turn, who moves, the scores and the bag.
export function PositionHeader({ p, children }: { p: BoardView; children?: ReactNode }) {
  return (
    <div className="dbp-header">
      <b>{p.title}</b> · turn {p.turn} · Player {p.mover + 1} to move · {p.scores[0]}–{p.scores[1]} · bag {p.bag_size}
      {children}
    </div>
  );
}

export function useKnownPositions(active: boolean): KnownPosition[] {
  const [positions, setPositions] = useState<KnownPosition[]>([]);
  // Retry until they load: at startup the data API can take ~10s to bind.
  useEffect(() => {
    if (positions.length > 0 || !active) return;
    let cancelled = false;
    const tryFetch = () =>
      getJSON('/api/known_positions')
        .then((d) => { if (!cancelled && d.positions.length) setPositions(d.positions); })
        .catch(() => {});
    tryFetch();
    const id = setInterval(tryFetch, 3000);
    return () => { cancelled = true; clearInterval(id); };
  }, [positions.length, active]);
  return positions;
}

export function PositionPicker({
  positions, index, onChange,
}: {
  positions: KnownPosition[];
  index: number;
  onChange: (i: number) => void;
}) {
  return (
    <div className="lane-controls-row">
      <label>
        Position{' '}
        <button className="arrow" onClick={() => onChange(Math.max(0, index - 1))}>◀</button>{' '}
        <select value={index} onChange={(e) => onChange(Number(e.target.value))}>
          {positions.map((p, i) => (
            <option key={p.id} value={i}>{p.id} — {p.title}</option>
          ))}
        </select>{' '}
        <button className="arrow" onClick={() => onChange(Math.min(positions.length - 1, index + 1))}>▶</button>
      </label>
    </div>
  );
}
