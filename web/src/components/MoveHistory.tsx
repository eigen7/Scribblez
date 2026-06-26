import React from 'react';

export interface TurnSummary {
  player: number;
  text: string;
  score: number;
  cumulative: [number, number];
}

// An end-of-game rack adjustment shown as a final row in the scoring player's
// column: the tiles scored/penalized, the point delta, and the resulting total.
export interface EndAdjustment {
  player: number;
  tiles: string;
  delta: number;
  total: number;
}

interface MoveHistoryProps {
  turns: TurnSummary[];
  playerNames: [string, string];
  endAdjustments?: EndAdjustment[];
  // Navigation (manual tool): when `onJumpToPly` is set, rows are clickable, the
  // viewed ply is highlighted, and a "Start Position" row is shown. Omitted for
  // a read-only live history (play_game).
  viewPly?: number;
  onJumpToPly?: (ply: number) => void;
}

// Split a turn's notation into the board location and main word shown as
// separate, aligned columns. A play reads as "<pos> <word> <score>" (three
// whitespace-free tokens); pass and exchange turns have no location, so their
// verb ("pass", "exch AQWW") fills the word cell.
function historyParts(text: string): { location: string; word: string } {
  if (/^(pass|exch)\b/.test(text)) {
    return { location: '', word: text };
  }
  const tokens = text.split(' ');
  return { location: tokens[0] ?? '', word: tokens[1] ?? '' };
}

function signed(n: number): string {
  return n >= 0 ? `+${n}` : `${n}`;
}

const MoveHistory: React.FC<MoveHistoryProps> = ({
  turns, playerNames, endAdjustments = [], viewPly, onJumpToPly,
}) => {
  const interactive = !!onJumpToPly;

  return (
    <div className="move-list manual-history-panel">
      <div className="move-list-header">
        <h3>Turn History ({turns.length})</h3>
      </div>
      <div className="manual-history">
        {interactive && (
          <button
            type="button"
            className={`manual-history-row manual-history-start${viewPly === 0 ? ' active' : ''}`}
            onClick={() => onJumpToPly?.(0)}
          >
            <span>Start Position</span>
            <span>0-0</span>
          </button>
        )}
        <div className="manual-history-columns">
          {[0, 1].map((p) => (
            <div key={p} className="manual-history-column">
              <div className="manual-history-name">{playerNames[p]}</div>
              {turns
                .map((t, idx) => ({ t, ply: idx + 1 }))
                .filter(({ t }) => t.player === p)
                .map(({ t, ply }) => {
                  const { location, word } = historyParts(t.text);
                  const active = viewPly === ply ? ' active' : '';
                  const cells = (
                    <>
                      <span className="mh-loc">{location}</span>
                      <span className="mh-word">{word}</span>
                      <span className="mh-delta">{signed(t.score)}</span>
                      <span className="mh-total">{t.cumulative[p]}</span>
                    </>
                  );
                  return interactive ? (
                    <button
                      key={ply}
                      type="button"
                      className={`manual-history-row manual-history-move${active}`}
                      onClick={() => onJumpToPly?.(ply)}
                    >
                      {cells}
                    </button>
                  ) : (
                    <div key={ply} className="manual-history-row manual-history-move manual-history-static">
                      {cells}
                    </div>
                  );
                })}
              {endAdjustments
                .filter((adj) => adj.player === p)
                .map((adj, i) => (
                  <div
                    key={`adj-${i}`}
                    className="manual-history-row manual-history-move manual-history-adjustment"
                  >
                    <span className="mh-loc" />
                    <span className="mh-word">({adj.tiles})</span>
                    <span className="mh-delta">{signed(adj.delta)}</span>
                    <span className="mh-total">{adj.total}</span>
                  </div>
                ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MoveHistory;
