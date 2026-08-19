import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { toBlob } from 'html-to-image';
import Board from './components/Board';
import Rack from './components/Rack';
import ScoreBoard from './components/ScoreBoard';
import { DragTilePayload, PlacedTile, TileInfo } from './types';

type ManualDirection = 'horizontal' | 'vertical';
// How much of the opponent's rack an exported GCG's #Rack lines reveal. The
// opponent is the player on turn at the latest position, whose tiles a position
// exported for study must not give away.
type GcgOpponentRack = 'full' | 'leave' | 'hidden';
type CandidateSource = { source: 'bag' } | { source: 'rack'; slot: number };
type ManualCandidate = PlacedTile & CandidateSource;

interface ManualRackSlot {
  letter: string;
  score: number;
  known: boolean;
  // Whether the player drew this tile after their own last move. Their leave
  // and their replenishment are shaded apart, so a reviewed position shows at a
  // glance which of the tiles they hold their last play accounts for.
  drawn: boolean;
  // Whether a tile occupies this slot at all. A present-but-unknown slot
  // renders as a "?" tile (selectable for exchange); an absent slot renders as
  // empty space.
  present: boolean;
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

// An end-of-game rack adjustment shown as a final history row in the scoring
// player's column: the tiles scored/penalized, the point delta, and the
// resulting total.
interface EndAdjustment {
  player: number;
  tiles: string;
  delta: number;
  total: number;
}

// Split a turn's notation into the board location and main word shown as
// separate, aligned columns in the history grid. A play reads as
// "<pos> <word> <score>" (three whitespace-free tokens); pass and exchange
// turns have no location, so their verb ("pass", "exch AQWW") fills the word
// cell and the location is left blank.
function historyParts(text: string): { location: string; word: string } {
  if (/^(pass|exch)\b/.test(text)) {
    return { location: '', word: text };
  }
  const tokens = text.split(' ');
  return { location: tokens[0] ?? '', word: tokens[1] ?? '' };
}

interface UnseenSlot {
  letter: string;
  present: boolean;
}

function rackSlotToTile(slot: ManualRackSlot): TileInfo {
  if (!slot.present) return { letter: '', score: 0, isAbsent: true };
  if (!slot.known) return { letter: '?', score: 0, isUnknown: true };
  if (slot.letter === '?') return { letter: '', score: 0, isBlank: true, isDrawn: slot.drawn };
  return { letter: slot.letter, score: slot.score, isDrawn: slot.drawn };
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
  end_adjustments: EndAdjustment[];
  last_move: [number, number][];
  view_ply: number;
  tail_ply: number;
  // The player on turn at the latest position: the "opponent" whose rack the
  // GCG export can mask.
  tail_player: 0 | 1;
  backtracking: boolean;
  game_over: boolean;
  status?: string;
  tile_scores: Record<string, number>;
}

const GCG_OPPONENT_RACK_CHOICES: ReadonlyArray<{
  value: GcgOpponentRack;
  label: string;
  hint: string;
}> = [
  { value: 'full', label: 'Full rack', hint: 'every tile, the ones drawn after their last move included' },
  { value: 'leave', label: 'Leave only', hint: 'what their last move kept; the tiles they drew stay "_"' },
  { value: 'hidden', label: 'Hidden', hint: 'no #Rack line for them at all' },
];

// How long to wait for the engine's answer to an export request before giving
// up on the save.
const GCG_REQUEST_TIMEOUT_MS = 5000;

interface PlayBuildResult {
  row: number;
  col: number;
  dir: ManualDirection;
  word: string;
}

const MAX_NAME_LENGTH = 18;

// Save a file via the browser's native "Save As" dialog (File System Access
// API), letting the user navigate the native filesystem. `produceBlob` runs
// only after the picker opens, so async rendering doesn't consume the click's
// user activation. Browsers without the API fall back to a normal download.
async function saveViaPicker(
  produceBlob: () => Promise<Blob>,
  suggestedName: string,
  accept: Record<string, string[]>,
  label: string,
  setStatus: (s: string) => void,
): Promise<void> {
  const picker = (
    window as unknown as {
      showSaveFilePicker?: (opts: object) => Promise<{
        createWritable: () => Promise<{
          write: (b: Blob) => Promise<void>;
          close: () => Promise<void>;
        }>;
      }>;
    }
  ).showSaveFilePicker;

  if (picker) {
    let handle;
    try {
      // A shared `id` makes the browser reopen in the last directory used for
      // either export; `startIn` is the first-time default (the Downloads dir).
      handle = await picker({ suggestedName, id: 'scribblez-save', startIn: 'downloads', types: [{ accept }] });
    } catch (e) {
      if ((e as { name?: string })?.name !== 'AbortError') setStatus('Failed to open save dialog.');
      return; // user cancelled
    }
    try {
      const blob = await produceBlob();
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      setStatus(`Saved ${label}.`);
    } catch {
      setStatus(`Failed to save ${label}.`);
    }
    return;
  }

  // Browsers without the File System Access API download to the default folder.
  try {
    const blob = await produceBlob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = suggestedName;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  } catch {
    setStatus(`Failed to export ${label}.`);
  }
}

function AppManual() {
  const [state, setState] = useState<ManualState | null>(null);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState('');
  // The game's title; also the default base name for GCG/PNG exports (with
  // characters illegal in filenames stripped).
  const [gameName, setGameName] = useState('');
  const exportBase = gameName.trim().replace(/[\\/:*?"<>|]/g, '') || 'Untitled';
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
  const [gcgExportOpen, setGcgExportOpen] = useState(false);
  const [gcgOpponentRack, setGcgOpponentRack] = useState<GcgOpponentRack>('full');
  const loadInputRef = useRef<HTMLInputElement | null>(null);
  const layoutRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  // The outstanding request for GCG text, resolved by the engine's reply.
  const gcgRequestRef = useRef<{
    resolve: (text: string) => void;
    reject: (error: Error) => void;
  } | null>(null);

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
      } else if (msg.type === 'manual_gcg') {
        gcgRequestRef.current?.resolve(msg.text ?? '');
      } else if (msg.type === 'manual_error') {
        gcgRequestRef.current?.reject(new Error(msg.message ?? 'backend error'));
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

  // Squares of the tiles placed by the move that produced the viewed position.
  const lastMoveCells = useMemo(
    () => new Set((state?.last_move ?? []).map(([r, c]) => `${r},${c}`)),
    [state?.last_move],
  );

  // Remaining count in the bag for each key (letter, or '?' for the blank),
  // after accounting for tiles already on the board, on known rack slots, and
  // staged as candidate plays. Used both to render the unseen pool and to block
  // placing a tile the bag can't supply.
  const bagRemaining = useMemo(() => {
    const remaining: Record<string, number> = {};
    if (!state) return remaining;
    const gone: Record<string, number> = {};
    for (const rack of state.racks) {
      for (const t of rack) {
        if (t.known) gone[t.letter] = (gone[t.letter] ?? 0) + 1;
      }
    }
    for (const row of state.board) {
      for (const cell of row) {
        if (!cell) continue;
        const key = cell >= 'a' && cell <= 'z' ? '?' : cell;
        gone[key] = (gone[key] ?? 0) + 1;
      }
    }
    for (const t of candidateTiles) {
      if (t.source !== 'bag') continue;
      const key = t.isBlank ? '?' : t.letter;
      gone[key] = (gone[key] ?? 0) + 1;
    }
    for (const [letter, total] of DISTRIBUTION) {
      remaining[letter] = Math.max(0, total - (gone[letter] ?? 0));
    }
    return remaining;
  }, [state, candidateTiles]);

  const bagBlocks = useCallback(
    (key: string): boolean => {
      if ((bagRemaining[key] ?? 0) > 0) return false;
      setStatus(key === '?' ? 'No blank tiles left in the bag' : `No ${key} tiles left in the bag`);
      return true;
    },
    [bagRemaining],
  );

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
    // A tile drawn from the bag can only be placed if the bag still has it.
    if (source.source === 'bag' && bagBlocks(payload.isBlank ? '?' : letter)) return;
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
      if (bagBlocks(isBlank ? '?' : letter)) return;
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
    bagBlocks,
  ]);

  // Left/right arrows step backward/forward through the move history. Skipped
  // while a text field is focused so they still move the caret there.
  useEffect(() => {
    const onArrowKey = (e: KeyboardEvent) => {
      if (!state) return;
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      const el = document.activeElement;
      if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) return;
      const target = e.key === 'ArrowLeft' ? state.view_ply - 1 : state.view_ply + 1;
      if (target < 0 || target > state.tail_ply) return;
      e.preventDefault();
      send({ type: 'jump_to_ply', ply: target });
      clearEntry();
    };
    window.addEventListener('keydown', onArrowKey);
    return () => window.removeEventListener('keydown', onArrowKey);
  }, [state, send, clearEntry]);

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

  // Ask the engine to serialize the game, revealing `opponentRack` of the
  // opponent's rack. Only one request is outstanding at a time.
  const requestGcg = useCallback(
    (opponentRack: GcgOpponentRack) =>
      new Promise<string>((resolve, reject) => {
        gcgRequestRef.current?.reject(new Error('superseded by another export'));
        const timer = window.setTimeout(() => {
          gcgRequestRef.current = null;
          reject(new Error('the engine did not answer'));
        }, GCG_REQUEST_TIMEOUT_MS);
        const settle = (finish: () => void) => {
          window.clearTimeout(timer);
          gcgRequestRef.current = null;
          finish();
        };
        gcgRequestRef.current = {
          resolve: (text) => settle(() => resolve(text)),
          reject: (error) => settle(() => reject(error)),
        };
        send({ type: 'export_gcg', opponent_rack: opponentRack });
      }),
    [send],
  );

  // Save the current game as GCG via the browser's native "Save As" dialog, so
  // the location is a native-filesystem path the user navigates to (like Load
  // Game's file chooser). The picker opens on this click, before the engine's
  // text is awaited, so the click's user activation still counts.
  const exportGcg = useCallback(() => {
    setGcgExportOpen(false);
    return saveViaPicker(
      async () => new Blob([await requestGcg(gcgOpponentRack)], { type: 'text/plain' }),
      `${exportBase}.gcg`,
      { 'text/plain': ['.gcg'] },
      'GCG',
      setStatus,
    );
  }, [requestGcg, gcgOpponentRack, exportBase]);

  // Render the board + racks + history + sidebar to a PNG matching the
  // on-screen look, and save it via the same native dialog. The page title is
  // outside the captured node; the action bars, status line, and cursor hint
  // are filtered out.
  const exportPng = useCallback(() => {
    const node = layoutRef.current;
    if (!node) return;
    return saveViaPicker(
      async () => {
        const blob = await toBlob(node, {
          pixelRatio: 2,
          backgroundColor: getComputedStyle(document.body).backgroundColor || '#1a2632',
          filter: (el: HTMLElement) =>
            !(
              el.classList?.contains('action-bar') ||
              el.classList?.contains('manual-status') ||
              el.classList?.contains('cursor-hint')
            ),
        });
        if (!blob) throw new Error('render failed');
        return blob;
      },
      `${exportBase}.png`,
      { 'image/png': ['.png'] },
      'PNG',
      setStatus,
    );
  }, [exportBase]);

  const newGame = () => {
    send({ type: 'reset' });
    clearEntry();
  };

  const createRandomGame = () => {
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

  const titleBar = (
    <h1 className="manual-title">
      Game:{' '}
      <input
        className="game-name-input"
        value={gameName}
        placeholder="[Untitled]"
        maxLength={60}
        onChange={(e) => setGameName(e.target.value)}
      />
    </h1>
  );

  if (!connected) {
    return (
      <div className="container">
        {titleBar}
        <p>Connecting to backend...</p>
        <button onClick={connect}>Reconnect</button>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="container">
        {titleBar}
        <p>Loading state...</p>
      </div>
    );
  }

  const unseenSlots: UnseenSlot[] = [];
  for (const [letter, total] of DISTRIBUTION) {
    const remain = bagRemaining[letter] ?? 0;
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
      {titleBar}
      <div className="manual-layout" ref={layoutRef}>
        <div className="manual-board-section">
          <div
            className={`manual-rack-row top${
              state.current_player === 1 ? ' manual-rack-row-active' : ''
            }`}
          >
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
            lastMoveCells={lastMoveCells}
          />

          <div
            className={`manual-rack-row${
              state.current_player === 0 ? ' manual-rack-row-active' : ''
            }`}
          >
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
                <button className="btn btn-pass" onClick={passTurn} disabled={state.game_over}>Pass</button>
                <button className="btn btn-exchange" onClick={startExchangeMode} disabled={state.game_over}>Exchange</button>
              </>
            )}
          </div>
          <div className="action-bar manual-secondary-actions">
            <button className="btn btn-game" onClick={newGame}>New Game</button>
            <button className="btn btn-game" onClick={createRandomGame}>Create Random Game</button>
          </div>
          <div className="action-bar manual-secondary-actions">
            <button className="btn btn-submit" onClick={forkGame}>Fork Game</button>
            <button className="btn btn-pass" onClick={triggerLoad}>Load Game</button>
            <button className="btn btn-exchange" onClick={() => setGcgExportOpen(true)}>Export GCG</button>
            <button className="btn btn-exchange" onClick={exportPng}>Export PNG</button>
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

          <div className="move-list manual-history-panel">
            <div className="move-list-header">
              <h3>Turn History ({state.turns.length})</h3>
            </div>
            <div className="manual-history">
              <button
                type="button"
                className={`manual-history-row manual-history-start${state.view_ply === 0 ? ' active' : ''}`}
                onClick={() => jumpToPly(0)}
              >
                <span>Start Position</span>
                <span>0-0</span>
              </button>
              <div className="manual-history-columns">
                {[0, 1].map((p) => (
                  <div key={p} className="manual-history-column">
                    <div className="manual-history-name">{state.player_names[p]}</div>
                    {state.turns
                      .map((t, idx) => ({ t, ply: idx + 1 }))
                      .filter(({ t }) => t.player === p)
                      .map(({ t, ply }) => {
                        const { location, word } = historyParts(t.text);
                        return (
                          <button
                            key={ply}
                            type="button"
                            className={`manual-history-row manual-history-move${
                              state.view_ply === ply ? ' active' : ''
                            }`}
                            onClick={() => jumpToPly(ply)}
                          >
                            <span className="mh-loc">{location}</span>
                            <span className="mh-word">{word}</span>
                            <span className="mh-delta">{t.score >= 0 ? `+${t.score}` : t.score}</span>
                            <span className="mh-total">{t.cumulative[p]}</span>
                          </button>
                        );
                      })}
                    {state.end_adjustments
                      .filter((adj) => adj.player === p)
                      .map((adj, i) => (
                        <div key={`adj-${i}`} className="manual-history-row manual-history-move manual-history-adjustment">
                          <span className="mh-loc" />
                          <span className="mh-word">({adj.tiles})</span>
                          <span className="mh-delta">{adj.delta >= 0 ? `+${adj.delta}` : adj.delta}</span>
                          <span className="mh-total">{adj.total}</span>
                        </div>
                      ))}
                  </div>
                ))}
              </div>
            </div>
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
        </div>
      </div>

      {gcgExportOpen && (
        <div className="modal-overlay" onClick={() => setGcgExportOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Export GCG</h3>
            <p className="gcg-export-note">
              How much of {state.player_names[state.tail_player]}'s rack &mdash; the player on turn
              &mdash; the file's #Rack lines reveal:
            </p>
            <div className="gcg-export-choices">
              {GCG_OPPONENT_RACK_CHOICES.map(({ value, label, hint }) => (
                <label key={value} className="gcg-export-choice">
                  <input
                    type="radio"
                    name="gcg-opponent-rack"
                    value={value}
                    checked={gcgOpponentRack === value}
                    onChange={() => setGcgOpponentRack(value)}
                  />
                  <span className="gcg-export-choice-label">{label}</span>
                  <span className="gcg-export-choice-hint">{hint}</span>
                </label>
              ))}
            </div>
            <div className="action-bar">
              <button className="btn btn-submit" onClick={exportGcg}>Export</button>
              <button className="btn btn-clear" onClick={() => setGcgExportOpen(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

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
