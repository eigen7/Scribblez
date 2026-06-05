import { useState, useEffect, useRef, useCallback } from 'react';
import Board from './components/Board';
import Rack from './components/Rack';
import MoveList from './components/MoveList';
import ScoreBoard from './components/ScoreBoard';
import UnseenTiles from './components/UnseenTiles';
import { GameState, PlacedTile, EntryDirection } from './types';
import {
  findRackTile,
  findMatchingMove,
  candidatesFromMove,
  rebuildUsedIndices as rebuildUsedIndicesPure,
} from './lib/moveMatch';

function App() {
  const [state, setState] = useState<GameState | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // Manual move entry state.
  const [candidateTiles, setCandidateTiles] = useState<PlacedTile[]>([]);
  const [usedRackIndices, setUsedRackIndices] = useState<Set<number>>(new Set());
  const [cursorRow, setCursorRow] = useState<number | null>(null);
  const [cursorCol, setCursorCol] = useState<number | null>(null);
  const [cursorDir, setCursorDir] = useState<EntryDirection | null>(null);

  // Cheat mode.
  const [cheatMode, setCheatMode] = useState(false);
  const [selectedMoveIndex, setSelectedMoveIndex] = useState<number | null>(null);

  // Exchange mode: clicking rack tiles toggles them into `exchangeSelected`,
  // and the action bar's confirm button sends them as a single exchange move.
  // Mutually exclusive with board entry (you can only enter exchange mode
  // when there are no candidate tiles on the board).
  const [exchangeMode, setExchangeMode] = useState(false);
  const [exchangeSelected, setExchangeSelected] = useState<Set<number>>(new Set());

  // Blank designation dialog.
  const [blankPending, setBlankPending] = useState<{ row: number; col: number; rackIndex: number } | null>(null);

  const connect = useCallback(() => {
    // Don't open a duplicate socket. React 18 StrictMode invokes effects
    // twice in dev (mount → unmount → mount); without this guard the second
    // mount would open a redundant connection.
    const existing = wsRef.current;
    if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      if (wsRef.current === ws) setConnected(true);
    };

    ws.onmessage = (event) => {
      const data: GameState = JSON.parse(event.data);
      setState(data);
      // Clear all entry state on new turn.
      clearEntry();
    };

    ws.onclose = () => {
      // Only react if this is still the active socket. A stale socket closing
      // (e.g. StrictMode's first, discarded mount) must not null out wsRef or
      // mark us disconnected while a newer socket is live -- otherwise every
      // move send would be silently dropped by submitMove's readyState guard.
      if (wsRef.current === ws) {
        setConnected(false);
        wsRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  const clearEntry = () => {
    setCandidateTiles([]);
    setUsedRackIndices(new Set());
    setCursorRow(null);
    setCursorCol(null);
    setCursorDir(null);
    setSelectedMoveIndex(null);
    setBlankPending(null);
    setExchangeMode(false);
    setExchangeSelected(new Set());
  };

  // --- Move submission ---

  const submitMove = (moveIndex: number) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: 'move', index: moveIndex }));
    setState((prev) => (prev ? { ...prev, your_turn: false } : prev));
    clearEntry();
  };

  const passTurn = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: 'pass' }));
    setState((prev) => (prev ? { ...prev, your_turn: false } : prev));
    clearEntry();
  };

  // Send the staged exchange. The backend accepts a `letters` string where
  // each char names one tile to discard ('?' for a blank); we just concat
  // the chosen rack tiles in their rack order.
  const submitExchange = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (!state) return;
    if (exchangeSelected.size === 0) return;
    const letters = state.rack
      .map((t, i) => (exchangeSelected.has(i) ? t.letter : ''))
      .join('');
    wsRef.current.send(JSON.stringify({ type: 'exchange', letters }));
    setState((prev) => (prev ? { ...prev, your_turn: false } : prev));
    clearEntry();
  };

  // Toggle one rack tile in/out of the exchange selection. No-op outside
  // exchange mode (the rack's interactive prop guards drag-to-board there).
  const handleRackTileClick = (index: number) => {
    if (!exchangeMode) return;
    setExchangeSelected((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const matchingMoveIndex = state
    ? findMatchingMove(candidateTiles, state.moves, state.board)
    : -1;

  // --- Board click handling ---

  const handleCellClick = (row: number, col: number) => {
    if (!state?.your_turn || state.game_over) return;
    if (state.board[row]?.[col]) return; // occupied

    // If there's a candidate tile here, remove it.
    const existingIdx = candidateTiles.findIndex((t) => t.row === row && t.col === col);
    if (existingIdx >= 0) {
      setCandidateTiles((prev) => prev.filter((_, i) => i !== existingIdx));
      // Find which rack index was used for this tile and free it.
      // We rebuild used indices from remaining candidates.
      rebuildUsedIndices(candidateTiles.filter((_, i) => i !== existingIdx));
      // Clear move list selection.
      setSelectedMoveIndex(null);

      // If this was the cursor position, keep it; otherwise set cursor here.
      if (cursorRow !== row || cursorCol !== col) {
        setCursorRow(row);
        setCursorCol(col);
        setCursorDir('horizontal');
      }
      return;
    }

    // Cursor logic: first click sets row (horizontal), second click on same cell sets column (vertical),
    // third click clears cursor.
    if (cursorRow === row && cursorCol === col) {
      if (cursorDir === 'horizontal') {
        setCursorDir('vertical');
      } else {
        setCursorRow(null);
        setCursorCol(null);
        setCursorDir(null);
      }
    } else {
      setCursorRow(row);
      setCursorCol(col);
      setCursorDir('horizontal');
    }

    // Clear move list selection when manually clicking.
    setSelectedMoveIndex(null);
  };

  // --- Keyboard entry ---

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!state?.your_turn || state.game_over) return;
      if (cursorRow === null || cursorCol === null || cursorDir === null) return;
      if (blankPending) return; // modal is open

      // Arrow keys move the highlighted (cursor) square. Left/right also set
      // the typing direction to horizontal, up/down to vertical, so subsequent
      // typing flows the way you just moved.
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' ||
          e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        e.preventDefault();
        if (e.key === 'ArrowLeft') {
          setCursorCol(Math.max(0, cursorCol - 1));
          setCursorDir('horizontal');
        } else if (e.key === 'ArrowRight') {
          setCursorCol(Math.min(14, cursorCol + 1));
          setCursorDir('horizontal');
        } else if (e.key === 'ArrowUp') {
          setCursorRow(Math.max(0, cursorRow - 1));
          setCursorDir('vertical');
        } else {
          setCursorRow(Math.min(14, cursorRow + 1));
          setCursorDir('vertical');
        }
        return;
      }

      if (e.key === 'Backspace' || e.key === 'Delete') {
        e.preventDefault();
        // Remove the last placed candidate tile.
        if (candidateTiles.length > 0) {
          const updated = candidateTiles.slice(0, -1);
          setCandidateTiles(updated);
          rebuildUsedIndices(updated);
          // Move cursor back.
          const last = candidateTiles[candidateTiles.length - 1];
          setCursorRow(last.row);
          setCursorCol(last.col);
        }
        setSelectedMoveIndex(null);
        return;
      }

      if (e.key === 'Escape') {
        e.preventDefault();
        clearEntry();
        return;
      }

      const letter = e.key.toUpperCase();
      if (!/^[A-Z]$/.test(letter)) return;
      e.preventDefault();

      // Find the next empty cell from cursor position.
      let r = cursorRow;
      let c = cursorCol;

      // Skip over occupied cells (board or candidates).
      while (r < 15 && c < 15) {
        if (!state.board[r]?.[c] && !candidateTiles.find((t) => t.row === r && t.col === c)) {
          break;
        }
        if (cursorDir === 'horizontal') c++;
        else r++;
      }
      if (r >= 15 || c >= 15) return;

      // Find matching rack tile (prefer non-blank).
      const rackIdx = findRackTile(state.rack, usedRackIndices, letter, false);
      if (rackIdx < 0) return;

      const isBlank = state.rack[rackIdx].letter === '?';
      const newTile: PlacedTile = { row: r, col: c, letter, isBlank };
      const newCandidates = [...candidateTiles, newTile];
      setCandidateTiles(newCandidates);

      const newUsed = new Set(usedRackIndices);
      newUsed.add(rackIdx);
      setUsedRackIndices(newUsed);

      // Advance cursor past this cell and any occupied cells.
      let nr = r;
      let nc = c;
      do {
        if (cursorDir === 'horizontal') nc++;
        else nr++;
      } while (nr < 15 && nc < 15 && (state.board[nr]?.[nc] || newCandidates.find((t) => t.row === nr && t.col === nc)));

      setCursorRow(nr < 15 ? nr : r);
      setCursorCol(nc < 15 ? nc : c);

      setSelectedMoveIndex(null);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state, cursorRow, cursorCol, cursorDir, candidateTiles, usedRackIndices, blankPending]);

  // --- Drag and drop ---

  const handleCellDrop = (row: number, col: number, letter: string, isBlank: boolean) => {
    if (!state?.your_turn || state.game_over) return;
    if (state.board[row]?.[col]) return;
    if (candidateTiles.find((t) => t.row === row && t.col === col)) return;

    if (isBlank) {
      // Get the rack index from the drag data — find first unused blank.
      const rackIdx = state.rack.findIndex((t, i) => !usedRackIndices.has(i) && t.letter === '?');
      if (rackIdx < 0) return;
      setBlankPending({ row, col, rackIndex: rackIdx });
      return;
    }

    const rackIdx = findRackTile(state.rack, usedRackIndices, letter, false);
    if (rackIdx < 0) return;

    const newTile: PlacedTile = { row, col, letter: letter.toUpperCase(), isBlank: false };
    setCandidateTiles((prev) => [...prev, newTile]);
    setUsedRackIndices((prev) => new Set([...prev, rackIdx]));
    setSelectedMoveIndex(null);
  };

  const handleBlankDesignation = (letter: string) => {
    if (!blankPending) return;
    const newTile: PlacedTile = {
      row: blankPending.row,
      col: blankPending.col,
      letter: letter.toUpperCase(),
      isBlank: true,
    };
    setCandidateTiles((prev) => [...prev, newTile]);
    setUsedRackIndices((prev) => new Set([...prev, blankPending.rackIndex]));
    setBlankPending(null);
    setSelectedMoveIndex(null);
  };

  // --- Cheat mode: selecting a move places its tiles as candidates ---

  const handleMoveSelect = (moveIndex: number) => {
    if (!state?.moves) return;
    const move = state.moves.find((m) => m.index === moveIndex);
    if (!move) return;

    if (selectedMoveIndex === moveIndex) {
      // Toggle off.
      clearEntry();
      return;
    }

    const result = candidatesFromMove(move, state.board, state.rack);
    if (!result) return;

    setCandidateTiles(result.candidates);
    setUsedRackIndices(result.usedRackIndices);
    setCursorRow(null);
    setCursorCol(null);
    setCursorDir(null);
    setSelectedMoveIndex(moveIndex);
  };

  // --- Helpers ---

  const rebuildUsedIndices = (tiles: PlacedTile[]) => {
    if (!state) return;
    setUsedRackIndices(rebuildUsedIndicesPure(tiles, state.rack));
  };

  // --- Render ---

  if (!connected) {
    return (
      <div className="container">
        <h1>Scribblez</h1>
        <p>Connecting to server...</p>
        <button onClick={connect}>Reconnect</button>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="container">
        <h1>Scribblez</h1>
        <p>Waiting for game to start...</p>
      </div>
    );
  }

  const canSubmit = matchingMoveIndex >= 0;

  return (
    <div className="container">
      <h1>Scribblez</h1>
      <div className="game-layout">
        <div className="board-section">
          <Board
            board={state.board}
            bonuses={state.bonuses}
            candidateTiles={candidateTiles}
            tileScores={state.tile_scores ?? {}}
            cursorRow={cursorRow}
            cursorCol={cursorCol}
            cursorDir={cursorDir}
            interactive={state.your_turn && !state.game_over}
            onCellClick={handleCellClick}
            onCellDrop={handleCellDrop}
          />
          <Rack
            tiles={state.rack}
            usedIndices={usedRackIndices}
            label="Your Rack"
            interactive={state.your_turn && !state.game_over}
            exchangeMode={exchangeMode}
            exchangeSelected={exchangeSelected}
            onTileClick={handleRackTileClick}
          />

          {/* Action buttons */}
          {state.your_turn && !state.game_over && (
            <div className="action-bar">
              {exchangeMode ? (
                <>
                  <button
                    className="btn btn-submit"
                    onClick={submitExchange}
                    disabled={exchangeSelected.size === 0}
                  >
                    Exchange {exchangeSelected.size} tile
                    {exchangeSelected.size === 1 ? '' : 's'}
                  </button>
                  <button
                    className="btn btn-clear"
                    onClick={() => {
                      setExchangeMode(false);
                      setExchangeSelected(new Set());
                    }}
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  {candidateTiles.length > 0 && (
                    <button className="btn btn-clear" onClick={clearEntry}>
                      Clear
                    </button>
                  )}
                  {canSubmit && (
                    <button
                      className="btn btn-submit"
                      onClick={() => submitMove(matchingMoveIndex)}
                    >
                      Play Move ({state.moves?.find((m) => m.index === matchingMoveIndex)?.score ?? 0} pts)
                    </button>
                  )}
                  {!canSubmit && candidateTiles.length > 0 && (
                    <span className="invalid-move-hint">Not a valid move</span>
                  )}
                  {candidateTiles.length === 0 && (
                    <>
                      <button className="btn btn-pass" onClick={passTurn}>
                        Pass
                      </button>
                      <button
                        className="btn btn-exchange"
                        onClick={() => setExchangeMode(true)}
                        disabled={state.bag_count < 7}
                        title={
                          state.bag_count < 7
                            ? 'Need at least 7 tiles in the bag to exchange'
                            : undefined
                        }
                      >
                        Exchange
                      </button>
                    </>
                  )}
                </>
              )}
            </div>
          )}

          {/* Cursor hint */}
          {cursorDir && cursorRow !== null && cursorCol !== null && (
            <div className="cursor-hint">
              Typing: {cursorDir === 'horizontal' ? '→' : '↓'} from {String.fromCharCode(65 + cursorCol)}{cursorRow + 1}
              {' '}(click cell to change direction, Esc to cancel)
            </div>
          )}
        </div>

        <div className="sidebar">
          <ScoreBoard
            playerNames={state.player_names}
            scores={state.scores}
            bagCount={state.bag_count}
            opponentRackCount={state.opponent_rack_count}
            yourTurn={state.your_turn}
            gameOver={state.game_over}
            winner={state.winner}
          />

          {/* Unseen tiles: bag + opponent rack, laid out as a 10x10 of the
              standard distribution; played/in-my-rack slots render blank. */}
          <UnseenTiles state={state} />

          {/* Cheat mode toggle */}
          <div className="cheat-toggle">
            <label>
              <input
                type="checkbox"
                checked={cheatMode}
                onChange={(e) => {
                  setCheatMode(e.target.checked);
                  if (!e.target.checked) clearEntry();
                }}
              />
              Show legal moves
            </label>
          </div>

          {cheatMode && state.moves && (
            <MoveList
              moves={state.moves}
              selectedIndex={selectedMoveIndex}
              onPreview={handleMoveSelect}
              disabled={!state.your_turn || state.game_over}
            />
          )}
        </div>
      </div>

      {/* Blank designation modal */}
      {blankPending && (
        <div className="modal-overlay" onClick={() => setBlankPending(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Choose letter for blank</h3>
            <div className="blank-letter-grid">
              {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').map((ch) => (
                <button
                  key={ch}
                  className="blank-letter-btn"
                  onClick={() => handleBlankDesignation(ch)}
                >
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

export default App;
