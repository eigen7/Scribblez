import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Board from './components/Board';
import Rack from './components/Rack';
import ScoreBoard from './components/ScoreBoard';
import { DragTilePayload, PlacedTile } from './types';

type ManualDirection = 'horizontal' | 'vertical';
type CandidateSource = { source: 'bag' } | { source: 'rack'; slot: number };
type ManualCandidate = PlacedTile & CandidateSource;

interface ManualRackSlot {
  letter: string;
  score: number;
  known: boolean;
}

interface BagTile {
  letter: string;
  count: number;
  score: number;
}

interface TurnSummary {
  player: number;
  text: string;
  score: number;
  cumulative: [number, number];
}

interface UnseenSlot {
  letter: string;
  present: boolean;
}

function rackSlotToTile(slot: ManualRackSlot): {
  letter: string;
  score: number;
  isBlank?: boolean;
  isUnknown?: boolean;
} {
  if (!slot.known) return { letter: '?', score: 0, isUnknown: true };
  if (slot.letter === '?') return { letter: '', score: 0, isBlank: true };
  return { letter: slot.letter, score: slot.score };
}

const DISTRIBUTION: ReadonlyArray<readonly [string, number]> = [
  ['A', 9], ['B', 2], ['C', 2], ['D', 4], ['E', 12], ['F', 2], ['G', 3], ['H', 2],
  ['I', 9], ['J', 1], ['K', 1], ['L', 4], ['M', 2], ['N', 6], ['O', 8], ['P', 2],
  ['Q', 1], ['R', 6], ['S', 4], ['T', 6], ['U', 4], ['V', 2], ['W', 2], ['X', 1],
  ['Y', 2], ['Z', 1], ['?', 2],
];

interface ManualState {
  type: 'manual_state';
  board: (string | null)[][];
  bonuses: (string | null)[][];
  scores: [number, number];
  player_names: [string, string];
  current_player: 0 | 1;
  racks: [ManualRackSlot[], ManualRackSlot[]];
  rack_known_counts: [number, number];
  bag_tiles: BagTile[];
  bag_count: number;
  lexicon: string;
  turns: TurnSummary[];
  view_ply: number;
  tail_ply: number;
  backtracking: boolean;
  status?: string;
  tile_scores: Record<string, number>;
}

interface PlayBuildResult {
  row: number;
  col: number;
  dir: ManualDirection;
  word: string;
}

const MAX_NAME_LENGTH = 18;

function AppManual() {
  const [state, setState] = useState<ManualState | null>(null);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState('');
  const [candidateTiles, setCandidateTiles] = useState<ManualCandidate[]>([]);
  const [cursorRow, setCursorRow] = useState<number | null>(null);
  const [cursorCol, setCursorCol] = useState<number | null>(null);
  const [cursorDir, setCursorDir] = useState<ManualDirection | null>(null);
  const [exchangeMode, setExchangeMode] = useState(false);
  const [exchangeSelected, setExchangeSelected] = useState<Set<number>>(new Set());
  const [rackSelection, setRackSelection] = useState<{ player: 0 | 1; slot: number } | null>(null);
  const [blankPending, setBlankPending] = useState<{
    row: number;
    col: number;
    source: CandidateSource;
  } | null>(null);
  const loadInputRef = useRef<HTMLInputElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const interactionLocked = !!state?.backtracking;

  const send = useCallback((payload: object) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify(payload));
  }, []);

  const clearEntry = useCallback(() => {
    setCandidateTiles([]);
    setCursorRow(null);
    setCursorCol(null);
    setCursorDir(null);
    setExchangeMode(false);
    setExchangeSelected(new Set());
    setBlankPending(null);
    setRackSelection(null);
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
      if (msg.type === 'manual_state') {
        setState(msg as ManualState);
        setStatus((msg as ManualState).status ?? '');
      } else if (msg.type === 'manual_error') {
        setStatus(msg.message ?? 'Unknown backend error.');
      }
    };

    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      wsRef.current = null;
      setConnected(false);
    };
  }, [send]);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  const candidateMap = useMemo(() => {
    const m = new Map<string, ManualCandidate>();
    for (const c of candidateTiles) m.set(`${c.row},${c.col}`, c);
    return m;
  }, [candidateTiles]);

  const onCellDrop = (row: number, col: number, payload: DragTilePayload) => {
    if (!state) return;
    if (interactionLocked) return;
    if (exchangeMode) return;
    if (state.board[row]?.[col]) return;
    if (candidateMap.has(`${row},${col}`)) return;
    if (payload.letter === '?' && !payload.isBlank) return;
    const letter = payload.letter.toUpperCase();
    const source: CandidateSource = payload.fromBag
      ? { source: 'bag' }
      : payload.rackIndex != null
        ? { source: 'rack', slot: payload.rackIndex }
        : { source: 'bag' };
    if (payload.isBlank) {
      setBlankPending({ row, col, source });
      return;
    }
    setCandidateTiles((prev) => [...prev, { row, col, letter, isBlank: !!payload.isBlank, ...source }]);
  };

  const handleCellClick = (row: number, col: number) => {
    if (!state) return;
    if (interactionLocked) return;
    if (exchangeMode) return;
    if (state.board[row]?.[col]) return;
    setRackSelection(null);
    const idx = candidateTiles.findIndex((t) => t.row === row && t.col === col);
    if (idx >= 0) {
      setCandidateTiles((prev) => prev.filter((_, i) => i !== idx));
      return;
    }
    if (cursorRow === row && cursorCol === col) {
      if (cursorDir === 'horizontal') setCursorDir('vertical');
      else {
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

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!state) return;
      if (interactionLocked) return;
      if (exchangeMode) return;
      if (blankPending) return;

      if (rackSelection) {
        const letter = e.key.toUpperCase();
        if (!/^[A-Z]$/.test(letter)) return;
        e.preventDefault();
        send({ type: 'set_rack_slot', player: rackSelection.player, slot: rackSelection.slot, letter });
        setRackSelection((prev) => {
          if (!prev) return prev;
          return { player: prev.player, slot: Math.min(6, prev.slot + 1) };
        });
        return;
      }

      if (cursorRow == null || cursorCol == null || cursorDir == null) return;

      if (e.key === 'Enter') {
        if (candidateTiles.length > 0) {
          e.preventDefault();
          submitMove();
        }
        return;
      }

      const letter = e.key.toUpperCase();
      if (!/^[A-Z]$/.test(letter)) {
        if (e.key === 'Backspace' || e.key === 'Delete') {
          e.preventDefault();
          if (candidateTiles.length > 0) {
            const last = candidateTiles[candidateTiles.length - 1];
            setCandidateTiles((prev) => prev.slice(0, -1));
            setCursorRow(last.row);
            setCursorCol(last.col);
          }
        }
        return;
      }
      e.preventDefault();

      let r = cursorRow;
      let c = cursorCol;
      while (r < 15 && c < 15) {
        if (!state.board[r]?.[c] && !candidateMap.has(`${r},${c}`)) break;
        if (cursorDir === 'horizontal') c += 1;
        else r += 1;
      }
      if (r >= 15 || c >= 15) return;

      const isBlank = e.shiftKey;
      setCandidateTiles((prev) => [...prev, { row: r, col: c, letter, isBlank, source: 'bag' }]);
      if (cursorDir === 'horizontal') setCursorCol(Math.min(14, c + 1));
      else setCursorRow(Math.min(14, r + 1));
    };

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    state,
    interactionLocked,
    exchangeMode,
    blankPending,
    rackSelection,
    cursorRow,
    cursorCol,
    cursorDir,
    candidateTiles,
    candidateMap,
    submitMove,
    send,
  ]);

  const handleBlankDesignation = (letter: string) => {
    if (interactionLocked) return;
    if (!blankPending) return;
    setCandidateTiles((prev) => [
      ...prev,
      {
        row: blankPending.row,
        col: blankPending.col,
        letter: letter.toUpperCase(),
        isBlank: true,
        ...blankPending.source,
      },
    ]);
    setBlankPending(null);
  };

  const handleRackSlotDrop = (player: number, index: number, payload: DragTilePayload) => {
    if (interactionLocked) return;
    if (!payload.fromBag) return;
    send({ type: 'set_rack_slot', player, slot: index, letter: payload.letter, isBlank: payload.isBlank });
  };

  const handleNameChange = (player: 0 | 1, name: string) => {
    if (interactionLocked) return;
    send({ type: 'set_name', player, name: name.slice(0, MAX_NAME_LENGTH) });
  };

  const handleRackTileClick = (index: number) => {
    if (!exchangeMode) return;
    setExchangeSelected((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const handleRackTileClickForPlayer = (player: 0 | 1, index: number) => {
    if (interactionLocked) return;
    if (exchangeMode) {
      if (player !== state?.current_player) return;
      handleRackTileClick(index);
      return;
    }
    setCursorRow(null);
    setCursorCol(null);
    setCursorDir(null);
    setRackSelection({ player, slot: index });
  };

  const buildPlay = (): PlayBuildResult | null => {
    if (!state) return null;
    if (candidateTiles.length === 0) return null;

    const sameRow = candidateTiles.every((t) => t.row === candidateTiles[0].row);
    const sameCol = candidateTiles.every((t) => t.col === candidateTiles[0].col);
    if (!sameRow && !sameCol) return null;
    const dir: ManualDirection = sameRow ? 'horizontal' : 'vertical';

    let row = candidateTiles[0].row;
    let col = candidateTiles[0].col;
    if (dir === 'horizontal') {
      const minCol = Math.min(...candidateTiles.map((t) => t.col));
      col = minCol;
      while (col > 0 && state.board[row]?.[col - 1]) col -= 1;
    } else {
      const minRow = Math.min(...candidateTiles.map((t) => t.row));
      row = minRow;
      while (row > 0 && state.board[row - 1]?.[col]) row -= 1;
    }

    const letters: string[] = [];
    let r = row;
    let c = col;
    while (r < 15 && c < 15) {
      const boardCell = state.board[r]?.[c];
      const cand = candidateMap.get(`${r},${c}`);
      if (!boardCell && !cand) break;
      if (boardCell) {
        letters.push(boardCell);
      } else if (cand) {
        letters.push(cand.isBlank ? cand.letter.toLowerCase() : cand.letter.toUpperCase());
      }
      if (dir === 'horizontal') c += 1;
      else r += 1;
    }

    if (letters.length === 0) return null;
    return { row, col, dir, word: letters.join('') };
  };

  function submitMove() {
    if (!state) return;
    if (interactionLocked) return;
    const built = buildPlay();
    if (!built) {
      setStatus('Candidate tiles must form one contiguous row or column.');
      return;
    }

    send({
      type: 'play',
      player: state.current_player,
      row: built.row,
      col: built.col,
      dir: built.dir,
      word: built.word,
      placements: candidateTiles.map((t) => ({
        row: t.row,
        col: t.col,
        letter: t.letter,
        isBlank: t.isBlank,
        source: t.source,
        slot: t.source === 'rack' ? t.slot : null,
      })),
    });
    clearEntry();
  }

  const passTurn = () => {
    if (!state) return;
    if (interactionLocked) return;
    send({ type: 'pass', player: state.current_player });
    clearEntry();
  };

  const submitExchange = () => {
    if (!state) return;
    if (interactionLocked) return;
    if (exchangeSelected.size === 0) {
      setStatus('Choose at least one rack slot to exchange.');
      return;
    }
    send({ type: 'exchange', player: state.current_player, slots: Array.from(exchangeSelected).sort((a, b) => a - b) });
    clearEntry();
  };

  const startExchangeMode = () => {
    if (interactionLocked) return;
    if (candidateTiles.length > 0) {
      setStatus('Clear candidate tiles before passing or exchanging.');
      return;
    }
    setExchangeMode(true);
    setExchangeSelected(new Set());
  };

  const exportGcg = () => {
    const defaultPath = '/workspace/repo/manual_positions.gcg';
    const path = window.prompt('Export path', defaultPath);
    if (!path) return;
    send({ type: 'export', path });
  };

  const newGame = () => {
    if (interactionLocked) return;
    send({ type: 'reset' });
    clearEntry();
  };

  const createRandomGame = () => {
    if (interactionLocked) return;
    send({ type: 'create_random_game' });
    clearEntry();
  };

  const jumpToPly = (ply: number) => {
    send({ type: 'jump_to_ply', ply });
    clearEntry();
  };

  const forkGame = () => {
    send({ type: 'fork_game' });
    clearEntry();
  };

  const triggerLoad = () => {
    loadInputRef.current?.click();
  };

  const onLoadFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      send({ type: 'load_gcg_text', text, file_name: file.name });
      clearEntry();
    } catch {
      setStatus(`Failed to read ${file.name}`);
    } finally {
      e.target.value = '';
    }
  };

  useEffect(() => {
    if (!state?.backtracking) return;
    clearEntry();
  }, [state?.backtracking, clearEntry]);

  if (!connected) {
    return (
      <div className="container">
        <h1>Scribblez Manual GCG Tool</h1>
        <p>Connecting to backend...</p>
        <button onClick={connect}>Reconnect</button>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="container">
        <h1>Scribblez Manual GCG Tool</h1>
        <p>Loading state...</p>
      </div>
    );
  }

  const knownRackCounts: Record<string, number> = {};
  for (const rack of state.racks) {
    for (const t of rack) {
      if (!t.known) continue;
      knownRackCounts[t.letter] = (knownRackCounts[t.letter] ?? 0) + 1;
    }
  }

  const boardCounts: Record<string, number> = {};
  for (const row of state.board) {
    for (const cell of row) {
      if (!cell) continue;
      const isBlank = cell >= 'a' && cell <= 'z';
      const key = isBlank ? '?' : cell;
      boardCounts[key] = (boardCounts[key] ?? 0) + 1;
    }
  }

  const candidateBagCounts: Record<string, number> = {};
  for (const t of candidateTiles) {
    if (t.source !== 'bag') continue;
    const key = t.isBlank ? '?' : t.letter;
    candidateBagCounts[key] = (candidateBagCounts[key] ?? 0) + 1;
  }

  const unseenSlots: UnseenSlot[] = [];
  for (const [letter, total] of DISTRIBUTION) {
    const gone = (knownRackCounts[letter] ?? 0) + (boardCounts[letter] ?? 0) + (candidateBagCounts[letter] ?? 0);
    const remain = Math.max(0, total - gone);
    for (let i = 0; i < remain; ++i) unseenSlots.push({ letter, present: true });
    for (let i = remain; i < total; ++i) unseenSlots.push({ letter, present: false });
  }
  const unseenCount = unseenSlots.reduce((n, s) => n + (s.present ? 1 : 0), 0);

  const bagDragStart = (e: React.DragEvent, tile: BagTile) => {
    if (interactionLocked) {
      e.preventDefault();
      return;
    }
    e.dataTransfer.setData(
      'text/plain',
      JSON.stringify({
        letter: tile.letter,
        isBlank: tile.letter === '?',
        score: tile.score,
        fromBag: true,
      } satisfies DragTilePayload),
    );
    e.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="container">
      <h1>Scribblez Manual GCG Tool</h1>
      <div className="manual-layout">
        <div className="manual-board-section">
          <div className="manual-rack-row top">
            <input
              className="rack-name-input"
              value={state.player_names[1]}
              maxLength={MAX_NAME_LENGTH}
              disabled={interactionLocked}
              onChange={(e) => handleNameChange(1, e.target.value)}
            />
            <Rack
              tiles={state.racks[1].map(rackSlotToTile)}
              usedIndices={new Set()}
              label=""
              interactive={!interactionLocked && state.current_player === 1}
              exchangeMode={exchangeMode && state.current_player === 1}
              exchangeSelected={exchangeSelected}
              onTileClick={(index) => handleRackTileClickForPlayer(1, index)}
              allowSlotDrop
              onTileDrop={(index, payload) => handleRackSlotDrop(1, index, payload)}
              showQuestionForBlank={false}
              hideScoreForQuestion
              draggableIndices={new Set(state.racks[1].map((t, i) => (t.known ? i : -1)).filter((i) => i >= 0))}
              tileClickEnabled={!exchangeMode && !interactionLocked}
              selectedTileIndex={rackSelection?.player === 1 ? rackSelection.slot : null}
            />
          </div>

          <Board
            board={state.board}
            bonuses={state.bonuses}
            candidateTiles={candidateTiles}
            tileScores={state.tile_scores}
            cursorRow={cursorRow}
            cursorCol={cursorCol}
            cursorDir={cursorDir}
            interactive={!interactionLocked}
            onCellClick={handleCellClick}
            onCellDrop={onCellDrop}
          />

          <div className="manual-rack-row">
            <input
              className="rack-name-input"
              value={state.player_names[0]}
              maxLength={MAX_NAME_LENGTH}
              disabled={interactionLocked}
              onChange={(e) => handleNameChange(0, e.target.value)}
            />
            <Rack
              tiles={state.racks[0].map(rackSlotToTile)}
              usedIndices={new Set()}
              label=""
              interactive={!interactionLocked && state.current_player === 0}
              exchangeMode={exchangeMode && state.current_player === 0}
              exchangeSelected={exchangeSelected}
              onTileClick={(index) => handleRackTileClickForPlayer(0, index)}
              allowSlotDrop
              onTileDrop={(index, payload) => handleRackSlotDrop(0, index, payload)}
              showQuestionForBlank={false}
              hideScoreForQuestion
              draggableIndices={new Set(state.racks[0].map((t, i) => (t.known ? i : -1)).filter((i) => i >= 0))}
              tileClickEnabled={!exchangeMode && !interactionLocked}
              selectedTileIndex={rackSelection?.player === 0 ? rackSelection.slot : null}
            />
          </div>

          <div className="action-bar">
            {interactionLocked ? (
              <>
                <button className="btn btn-clear" onClick={() => jumpToPly(state.tail_ply)}>
                  Return to End
                </button>
              </>
            ) : exchangeMode ? (
              <>
                <button className="btn btn-submit" onClick={submitExchange}>
                  Exchange {exchangeSelected.size}
                </button>
                <button className="btn btn-clear" onClick={clearEntry}>Cancel</button>
              </>
            ) : candidateTiles.length > 0 ? (
              <>
                <button className="btn btn-clear" onClick={clearEntry}>Clear</button>
                <button className="btn btn-submit" onClick={submitMove}>Play Move</button>
              </>
            ) : (
              <>
                <button className="btn btn-pass" onClick={passTurn}>Pass</button>
                <button className="btn btn-exchange" onClick={startExchangeMode}>Exchange</button>
              </>
            )}
          </div>
          <div className="action-bar manual-secondary-actions">
            <button className="btn btn-clear" onClick={newGame} disabled={interactionLocked}>New Game</button>
            <button className="btn btn-clear" onClick={createRandomGame} disabled={interactionLocked}>Create Random Game</button>
            {interactionLocked && (
              <button className="btn btn-submit" onClick={forkGame}>Fork Game</button>
            )}
            <button className="btn btn-pass" onClick={triggerLoad}>Load Game</button>
            <button className="btn btn-exchange" onClick={exportGcg}>Export GCG</button>
            <input
              ref={loadInputRef}
              type="file"
              accept=".gcg,text/plain"
              style={{ display: 'none' }}
              onChange={onLoadFileSelected}
            />
          </div>
          {status && <div className="manual-status">{status}</div>}
          <div className="cursor-hint">
            Lexicon: {state.lexicon} | Rack known: {state.rack_known_counts[0]} / {state.rack_known_counts[1]}
          </div>
        </div>

        <div className="sidebar">
          <ScoreBoard
            playerNames={state.player_names}
            scores={state.scores}
            bagCount={state.bag_count}
            opponentRackCount={0}
            rackCounts={state.rack_known_counts}
            yourTurn={state.current_player === 0}
            gameOver={false}
            editableNames
            maxNameLength={MAX_NAME_LENGTH}
            onNameChange={handleNameChange}
            showTurnIndicator={false}
          />

          <div className="move-list">
            <div className="move-list-header">
              <h3>Unseen ({unseenCount})</h3>
            </div>
            <div className="unseen-tiles-grid">
              {unseenSlots.map((slot, i) => (
                slot.present ? (
                  <div
                    key={i}
                    className="unseen-cell present manual-unseen-draggable"
                    draggable={!interactionLocked}
                    onDragStart={(e) => bagDragStart(e, { letter: slot.letter, count: 1, score: slot.letter === '?' ? 0 : (state.tile_scores[slot.letter] ?? 0) })}
                  >
                    <span>{slot.letter === '?' ? '' : slot.letter}</span>
                  </div>
                ) : (
                  <div key={i} className="unseen-cell absent" />
                )
              ))}
            </div>
          </div>

          <div className="move-list">
            <div className="move-list-header">
              <h3>Turn History ({state.turns.length})</h3>
            </div>
            <div className="manual-history">
              {state.turns
                .slice()
                .map((t, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className={`manual-history-row${state.view_ply === idx + 1 ? ' active' : ''}`}
                    onClick={() => jumpToPly(idx + 1)}
                  >
                    <span>{state.player_names[t.player]}: {t.text}</span>
                    <span>{t.cumulative[0]}-{t.cumulative[1]}</span>
                  </button>
                ))}
              <button
                type="button"
                className={`manual-history-row manual-history-start${state.view_ply === 0 ? ' active' : ''}`}
                onClick={() => jumpToPly(0)}
              >
                <span>Start Position</span>
                <span>0-0</span>
              </button>
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

export default AppManual;
