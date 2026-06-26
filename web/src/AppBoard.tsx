import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Board from './components/Board';
import { DragTilePayload } from './types';

type BoardDirection = 'horizontal' | 'vertical';

interface BagTile {
  letter: string;
  count: number;
  score: number;
}

interface InvalidWord {
  word: string;
  square: string;
  direction: string;
}

interface BoardValidation {
  valid: boolean;
  invalid_words: InvalidWord[];
}

interface BoardToolState {
  type: 'board_state';
  board: (string | null)[][];
  bonuses: (string | null)[][];
  bag_tiles: BagTile[];
  bag_count: number;
  tile_scores: Record<string, number>;
  lexicon: string;
  status?: string;
  validation: BoardValidation | null;
}

// Full English Scrabble tile distribution; '?' is the blank. Used to render the
// bag grid with depleted slots greyed out.
const DISTRIBUTION: ReadonlyArray<readonly [string, number]> = [
  ['A', 9], ['B', 2], ['C', 2], ['D', 4], ['E', 12], ['F', 2], ['G', 3], ['H', 2],
  ['I', 9], ['J', 1], ['K', 1], ['L', 4], ['M', 2], ['N', 6], ['O', 8], ['P', 2],
  ['Q', 1], ['R', 6], ['S', 4], ['T', 6], ['U', 4], ['V', 2], ['W', 2], ['X', 1],
  ['Y', 2], ['Z', 1], ['?', 2],
];

function AppBoard() {
  const [state, setState] = useState<BoardToolState | null>(null);
  const [status, setStatus] = useState('');
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const [cursorRow, setCursorRow] = useState<number | null>(null);
  const [cursorCol, setCursorCol] = useState<number | null>(null);
  const [cursorDir, setCursorDir] = useState<BoardDirection | null>(null);
  const [blankPending, setBlankPending] = useState<{ row: number; col: number } | null>(null);

  const send = useCallback((payload: object) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify(payload));
  }, []);

  const connect = useCallback(() => {
    const existing = wsRef.current;
    if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
      return;
    }
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    wsRef.current = ws;
    ws.onopen = () => {
      if (wsRef.current !== ws) return;
      setConnected(true);
    };
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'board_state') {
        setState(msg as BoardToolState);
        setStatus((msg as BoardToolState).status ?? '');
      }
    };
    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      wsRef.current = null;
      setConnected(false);
    };
  }, []);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  // Remaining bag count per key (letter, or '?' for the blank). Tiles commit to
  // the board immediately, so the backend bag count already reflects the board.
  const bagCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const t of state?.bag_tiles ?? []) counts[t.letter] = t.count;
    return counts;
  }, [state?.bag_tiles]);

  const placeTile = useCallback(
    (row: number, col: number, letter: string, isBlank: boolean): boolean => {
      const key = isBlank ? '?' : letter;
      if ((bagCounts[key] ?? 0) <= 0) {
        setStatus(isBlank ? 'No blank tiles left in the bag' : `No ${letter} tiles left in the bag`);
        return false;
      }
      send({ type: 'place', row, col, letter, isBlank });
      return true;
    },
    [bagCounts, send],
  );

  const handleCellClick = (row: number, col: number) => {
    if (!state) return;
    // A filled square is cleared on click; an empty square sets/toggles the
    // keyboard cursor.
    if (state.board[row]?.[col]) {
      send({ type: 'remove', row, col });
      return;
    }
    if (cursorRow === row && cursorCol === col) {
      if (cursorDir === 'horizontal') {
        setCursorDir('vertical');
      } else {
        setCursorRow(null);
        setCursorCol(null);
        setCursorDir(null);
      }
      return;
    }
    setCursorRow(row);
    setCursorCol(col);
    setCursorDir('horizontal');
  };

  const onCellDrop = (row: number, col: number, payload: DragTilePayload) => {
    if (!state) return;
    if (state.board[row]?.[col]) return;
    if (payload.letter === '?' && !payload.isBlank) return;
    if (payload.isBlank) {
      setBlankPending({ row, col });
      return;
    }
    placeTile(row, col, payload.letter.toUpperCase(), false);
  };

  const handleBlankDesignation = (letter: string) => {
    if (!blankPending) return;
    placeTile(blankPending.row, blankPending.col, letter.toUpperCase(), true);
    setBlankPending(null);
  };

  // Keyboard entry: type a letter to drop a tile at the cursor and advance;
  // shift makes it a blank. Backspace clears the previous square.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!state) return;
      if (blankPending) return;
      const el = document.activeElement;
      if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) return;
      if (cursorRow == null || cursorCol == null || cursorDir == null) return;

      const stepBack = () => {
        if (cursorDir === 'horizontal') setCursorCol((c) => (c == null ? c : Math.max(0, c - 1)));
        else setCursorRow((r) => (r == null ? r : Math.max(0, r - 1)));
      };

      if (e.key === 'Backspace' || e.key === 'Delete') {
        e.preventDefault();
        let r = cursorRow;
        let c = cursorCol;
        if (cursorDir === 'horizontal') c = Math.max(0, c - 1);
        else r = Math.max(0, r - 1);
        if (state.board[r]?.[c]) send({ type: 'remove', row: r, col: c });
        stepBack();
        return;
      }

      const letter = e.key.toUpperCase();
      if (!/^[A-Z]$/.test(letter)) return;
      e.preventDefault();

      // Advance to the next empty square from the cursor.
      let r = cursorRow;
      let c = cursorCol;
      while (r < 15 && c < 15) {
        if (!state.board[r]?.[c]) break;
        if (cursorDir === 'horizontal') c += 1;
        else r += 1;
      }
      if (r >= 15 || c >= 15) return;

      if (!placeTile(r, c, letter, e.shiftKey)) return;
      if (cursorDir === 'horizontal') setCursorCol(Math.min(14, c + 1));
      else setCursorRow(Math.min(14, r + 1));
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [state, blankPending, cursorRow, cursorCol, cursorDir, placeTile, send]);

  const bagDragStart = (e: React.DragEvent, letter: string) => {
    e.dataTransfer.setData(
      'text/plain',
      JSON.stringify({
        letter,
        isBlank: letter === '?',
        score: letter === '?' ? 0 : (state?.tile_scores[letter] ?? 0),
        fromBag: true,
      } satisfies DragTilePayload),
    );
    e.dataTransfer.effectAllowed = 'move';
  };

  if (!connected || !state) {
    return (
      <div className="container">
        <h1>Scribblez Board Tool</h1>
        <p>{connected ? 'Loading…' : 'Connecting…'}</p>
      </div>
    );
  }

  const validation = state.validation;

  return (
    <div className="container">
      <h1>Scribblez Board Tool</h1>
      <div className="manual-layout">
        <div className="manual-board-section">
          <Board
            board={state.board}
            bonuses={state.bonuses}
            candidateTiles={[]}
            tileScores={state.tile_scores}
            cursorRow={cursorRow}
            cursorCol={cursorCol}
            cursorDir={cursorDir}
            interactive
            onCellClick={handleCellClick}
            onCellDrop={onCellDrop}
          />

          <div className="action-bar manual-secondary-actions">
            <button className="btn btn-submit" onClick={() => send({ type: 'validate' })}>
              Validate
            </button>
          </div>

          {validation && (
            <div className={`board-validation ${validation.valid ? 'valid' : 'invalid'}`}>
              {validation.valid ? (
                <span>✓ The board is valid — every word is legal in {state.lexicon}.</span>
              ) : (
                <>
                  <div>✗ {validation.invalid_words.length} invalid word
                    {validation.invalid_words.length === 1 ? '' : 's'}:</div>
                  <ul>
                    {validation.invalid_words.map((w, i) => (
                      <li key={i}>
                        <strong>{w.word}</strong> — {w.square} {w.direction}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )}

          {status && <div className="manual-status">{status}</div>}
          <div className="cursor-hint">
            Lexicon: {state.lexicon} | Bag: {state.bag_count} | Drag tiles from the bag or click a
            square and type. Shift-type or drag a blank for a blank tile. Click a placed tile to
            remove it.
          </div>
        </div>

        <div className="sidebar">
          <div className="move-list">
            <div className="move-list-header">
              <h3>Bag ({state.bag_count})</h3>
            </div>
            <div className="unseen-tiles-grid">
              {DISTRIBUTION.flatMap(([letter, total]) => {
                const present = bagCounts[letter] ?? 0;
                return Array.from({ length: total }, (_, i) =>
                  i < present ? (
                    <div
                      key={`${letter}-${i}`}
                      className="unseen-cell present manual-unseen-draggable"
                      draggable
                      onDragStart={(e) => bagDragStart(e, letter)}
                    >
                      <span>{letter === '?' ? '?' : letter}</span>
                    </div>
                  ) : (
                    <div key={`${letter}-${i}`} className="unseen-cell absent" />
                  ),
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {blankPending && (
        <div className="modal-overlay" onClick={() => setBlankPending(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Choose letter for blank</h3>
            <div className="blank-letter-grid">
              {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').map((ch) => (
                <button key={ch} className="blank-letter-btn" onClick={() => handleBlankDesignation(ch)}>
                  {ch}
                </button>
              ))}
            </div>
            <button className="btn btn-clear" onClick={() => setBlankPending(null)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default AppBoard;
