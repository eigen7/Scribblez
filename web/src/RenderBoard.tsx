import { useEffect, useRef } from 'react';
import Board from './components/Board';
import Rack from './components/Rack';
import { ManualRackSlot, rackSlotToTile } from './manualState';
import { captureLayoutDataUrl } from './exportImage';

// The subset of the engine's `manual_state` this offline harness draws. Injected
// on `window` by the render driver (see web/scripts/render_boards.mjs) before the
// page loads; the driver reads the captured PNG back off `window` afterward.
interface RenderState {
  board: (string | null)[][];
  bonuses: (string | null)[][];
  racks: [ManualRackSlot[], ManualRackSlot[]];
  player_names: [string, string];
  scores: [number, number];
  tile_scores: Record<string, number>;
  // [row, col] squares of the move that produced this position, highlighted.
  last_move?: [number, number][];
  // Optional caption drawn under the board (e.g. "Nigel to play, up 53").
  caption?: string;
}

declare global {
  interface Window {
    __RENDER_STATE__?: RenderState;
    __RENDER_PNG__?: string;
    __RENDER_ERROR__?: string;
  }
}

const NOOP = () => {};

// A read-only seat header: the player's name and current score, drawn where the
// live UI puts its editable name field.
function SeatLabel({ name, score }: { name: string; score: number }) {
  return (
    <div className="render-seat-label">
      <span className="render-seat-name">{name}</span>
      <span className="render-seat-score">{score}</span>
    </div>
  );
}

// One non-interactive rack row (seat label + tiles), matching the live UI's
// `.manual-rack-row` layout so the captured image lines up with the board.
function RackRow({
  slots,
  name,
  score,
  top,
}: {
  slots: ManualRackSlot[];
  name: string;
  score: number;
  top: boolean;
}) {
  return (
    <div className={`manual-rack-row${top ? ' top' : ''}`}>
      <SeatLabel name={name} score={score} />
      <Rack
        tiles={slots.map(rackSlotToTile)}
        usedIndices={new Set()}
        label=""
        interactive={false}
        onTileClick={NOOP}
        showQuestionForBlank={false}
        hideScoreForQuestion
      />
    </div>
  );
}

// Headless board renderer: draws the injected position exactly as the manual UI
// would, then captures it to a PNG data URL via the shared Export-PNG path and
// hands it back on `window.__RENDER_PNG__`. Reached at `?tool=render`.
export default function RenderBoard() {
  const layoutRef = useRef<HTMLDivElement | null>(null);
  const state = window.__RENDER_STATE__;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        if (!state) throw new Error('window.__RENDER_STATE__ is not set');
        // Let fonts and layout settle so the capture matches the on-screen look.
        await document.fonts.ready;
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        if (cancelled || !layoutRef.current) return;
        window.__RENDER_PNG__ = await captureLayoutDataUrl(layoutRef.current);
      } catch (err) {
        window.__RENDER_ERROR__ = err instanceof Error ? err.message : String(err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [state]);

  if (!state) return <div className="render-missing">No render state provided.</div>;

  const lastMoveCells = new Set((state.last_move ?? []).map(([r, c]) => `${r},${c}`));

  return (
    <div className="render-root">
      <div className="manual-board-section" ref={layoutRef}>
        <RackRow slots={state.racks[1]} name={state.player_names[1]} score={state.scores[1]} top />
        <Board
          board={state.board}
          bonuses={state.bonuses}
          candidateTiles={[]}
          tileScores={state.tile_scores}
          cursorRow={null}
          cursorCol={null}
          cursorDir={null}
          interactive={false}
          onCellClick={NOOP}
          onCellDrop={NOOP}
          lastMoveCells={lastMoveCells}
        />
        <RackRow slots={state.racks[0]} name={state.player_names[0]} score={state.scores[0]} top={false} />
        {state.caption && <div className="render-caption">{state.caption}</div>}
      </div>
    </div>
  );
}
