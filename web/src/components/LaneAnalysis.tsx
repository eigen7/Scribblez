import { useCallback, useEffect, useState } from 'react';
import Board from './Board';
import { getJSON } from '../lib/api';

// The score head has 100 bins; the top bin is the catch-all for scores >= 99.
const TOP_SCORE_BIN = 99;

interface BestMove {
  word: string;
  row: number;
  col: number;
  horizontal: boolean;
  score: number;
}

interface Lane {
  has_move: boolean;
  max_score: number;
  num_best: number;
  placed: string[][]; // per-cell true union (letters)
  best_moves: BestMove[];
  pred_placed?: string[][];
  pred_score_bin?: number;
  pred_score_pmf?: number[];
  pred_has_move?: number;
  move_correct?: boolean;
  score_correct?: boolean;
}

interface Payload {
  name: string;
  on_move: number;
  board: (string | null)[][];
  bonuses: (string | null)[][];
  tile_scores: Record<string, number>;
  generation: number | null;
  has_prediction: boolean;
  lanes: { rows: Lane[]; cols: Lane[] };
}

interface Generation {
  generation: number;
  positions: number;
}

// The per-cell true-vs-predicted union comparison for one lane.
function diffCells(lane: Lane): { t: string[]; p: string[]; status: string }[] {
  const pred = lane.pred_placed ?? Array.from({ length: 15 }, () => []);
  return lane.placed.map((t, i) => {
    const p = pred[i] ?? [];
    const ps = new Set(p);
    const ts = new Set(t);
    const eq = t.length === p.length && t.every((x) => ps.has(x));
    let status = 'neutral';
    if (t.length || p.length) {
      if (eq) status = 'match';
      else if (p.some((x) => !ts.has(x))) status = 'extra'; // model placed a tile the truth doesn't
      else status = 'missed'; // truth has a tile the model didn't place
    }
    return { t, p, status };
  });
}

function LaneStatusCell({ lane, selected, onClick }: { lane: Lane; selected: boolean; onClick: () => void }) {
  if (!lane.has_move) return <div className="lane-cell empty" onClick={onClick} />;
  return (
    <div className={`lane-cell${selected ? ' selected' : ''}`} onClick={onClick}>
      <span className={`lane-mark move${lane.move_correct ? ' ok' : ''}`} title="best move" />
      <span className={`lane-mark score${lane.score_correct ? ' ok' : ''}`} title="best score" />
    </div>
  );
}

function ScoreHistogram({ lane }: { lane: Lane }) {
  const pmf = lane.pred_score_pmf;
  if (!pmf) return <div style={{ color: '#889', fontStyle: 'italic' }}>No prediction.</div>;
  const max = Math.max(...pmf, 1e-9);
  const truth = Math.min(lane.max_score, TOP_SCORE_BIN);
  return (
    <div>
      <div className="score-hist">
        {pmf.map((v, i) => (
          <div
            key={i}
            className={`hbar${i === truth ? ' truth' : ''}`}
            style={{ height: `${(v / max) * 100}%` }}
            title={`score ${i}: ${(v * 100).toFixed(1)}%`}
          />
        ))}
      </div>
      <div className="legend">
        true score <b>{truth}</b>
        <span className="sw" style={{ background: '#e67e22' }} />
        predicted argmax <b>{lane.pred_score_bin}</b>
        <span className="sw" style={{ background: '#5dade2' }} />
      </div>
    </div>
  );
}

export default function LaneAnalysis({ task, tag }: { task: string; tag: string | null }) {
  const [positions, setPositions] = useState<string[]>([]);
  const [generations, setGenerations] = useState<Generation[]>([]);
  const [posIdx, setPosIdx] = useState(0);
  const [genIdx, setGenIdx] = useState(0); // index into `generations` when not following latest
  const [latest, setLatest] = useState(true);
  const [horizontal, setHorizontal] = useState(true);
  const [laneIdx, setLaneIdx] = useState(7);
  const [payload, setPayload] = useState<Payload | null>(null);

  // The dataset's positions (fixed). Retry until they load: at startup the data
  // API can take ~10s to bind (it imports torch + the engine FFI), so the first
  // fetch may fail; without a retry the tab would stay empty permanently.
  useEffect(() => {
    if (positions.length > 0) return;
    let cancelled = false;
    const tryFetch = () =>
      getJSON('/api/lane/positions')
        .then((d) => {
          if (!cancelled && d.positions.length) setPositions(d.positions);
        })
        .catch(() => {});
    tryFetch();
    const id = setInterval(tryFetch, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [positions.length]);

  const refreshGens = useCallback(() => {
    if (!tag) return;
    getJSON(`/api/lane/generations?task=${task}&tag=${tag}`)
      .then((d) => setGenerations(d.generations))
      .catch(() => {});
  }, [task, tag]);
  useEffect(refreshGens, [refreshGens]);
  useEffect(() => {
    const id = setInterval(refreshGens, 3000);
    return () => clearInterval(id);
  }, [refreshGens]);

  const genCount = generations.length;
  const effIdx = latest ? Math.max(0, genCount - 1) : Math.min(genIdx, Math.max(0, genCount - 1));
  const effGen = generations[effIdx]?.generation ?? null;

  // Fetch the merged board + ground-truth + prediction payload.
  const fetchPayload = useCallback(() => {
    if (!tag || positions.length === 0) {
      setPayload(null);
      return;
    }
    const g = effGen == null ? 'latest' : String(effGen);
    getJSON(`/api/lane/position?task=${task}&tag=${tag}&position=${posIdx}&generation=${g}`)
      .then(setPayload)
      .catch(() => setPayload(null));
  }, [task, tag, positions.length, posIdx, effGen]);
  useEffect(() => {
    fetchPayload();
  }, [fetchPayload]);

  if (!tag) return <div style={{ color: '#889', padding: 20 }}>Select a tag.</div>;
  if (positions.length === 0) {
    return <div style={{ color: '#889', padding: 20 }}>No lane-analysis dataset available.</div>;
  }

  const axis = payload ? (horizontal ? payload.lanes.rows : payload.lanes.cols) : [];
  const lane: Lane | undefined = axis[laneIdx];

  const selectLane = (h: boolean, i: number) => {
    setHorizontal(h);
    setLaneIdx(i);
  };

  return (
    <div>
      {/* Controls */}
      <div className="lane-controls">
        <label>
          Model{' '}
          <input
            type="range"
            min={0}
            max={Math.max(0, genCount - 1)}
            value={effIdx}
            disabled={latest || genCount <= 1}
            onChange={(e) => setGenIdx(Number(e.target.value))}
          />{' '}
          <span style={{ color: '#667' }}>
            {effGen == null ? 'no checkpoints yet' : `gen ${effGen} (${generations[effIdx]?.positions.toLocaleString()} pos)`}
          </span>
        </label>
        <label>
          <input type="checkbox" checked={latest} onChange={(e) => setLatest(e.target.checked)} /> latest
        </label>

        <label>
          Position{' '}
          <button className="arrow" onClick={() => setPosIdx((i) => Math.max(0, i - 1))}>
            ◀
          </button>{' '}
          <select value={posIdx} onChange={(e) => setPosIdx(Number(e.target.value))}>
            {positions.map((name, i) => (
              <option key={name} value={i}>
                {name}
              </option>
            ))}
          </select>{' '}
          <button className="arrow" onClick={() => setPosIdx((i) => Math.min(positions.length - 1, i + 1))}>
            ▶
          </button>
        </label>

        <span className="seg">
          <button className={horizontal ? 'active' : ''} onClick={() => setHorizontal(true)}>
            Horizontal
          </button>
          <button className={!horizontal ? 'active' : ''} onClick={() => setHorizontal(false)}>
            Vertical
          </button>
        </span>
      </div>

      {!payload ? (
        <div style={{ color: '#889', padding: 20 }}>Loading position…</div>
      ) : (
        <>
          {/* Board flanked by per-lane status strips. */}
          <div className="lane-board-area">
            <Board
              board={payload.board}
              bonuses={payload.bonuses}
              candidateTiles={[]}
              tileScores={payload.tile_scores}
              cursorRow={null}
              cursorCol={null}
              cursorDir={null}
              interactive
              onCellClick={(r, c) => setLaneIdx(horizontal ? r : c)}
              onCellDrop={() => {}}
              highlightLane={{ horizontal, index: laneIdx }}
            />
            <div className="lane-row-strip">
              {payload.lanes.rows.map((l, r) => (
                <LaneStatusCell
                  key={r}
                  lane={l}
                  selected={horizontal && laneIdx === r}
                  onClick={() => selectLane(true, r)}
                />
              ))}
            </div>
            <div className="lane-col-strip">
              {payload.lanes.cols.map((l, c) => (
                <LaneStatusCell
                  key={c}
                  lane={l}
                  selected={!horizontal && laneIdx === c}
                  onClick={() => selectLane(false, c)}
                />
              ))}
            </div>
          </div>

          {/* Lane detail. */}
          <div className="lane-detail">
            <h3>
              {horizontal ? 'Row' : 'Column'} {laneIdx + 1}
              {lane && !lane.has_move && ' — no legal move'}
            </h3>
            {lane && lane.has_move && (
              <>
                <div className="best-moves">
                  True best (score {lane.max_score}
                  {lane.num_best > 1 ? `, ${lane.num_best} tied` : ''}):{' '}
                  {lane.best_moves.map((m, i) => (
                    <span className="bm" key={i}>
                      <b>{m.word}</b> @ {String.fromCharCode(65 + m.col)}
                      {m.row + 1}
                    </span>
                  ))}
                </div>

                <div className="union-diff">
                  {diffCells(lane).map((d, i) => (
                    <div className={`dc ${d.status}`} key={i} title={`cell ${i + 1}`}>
                      <span className="t">{d.t.join('') || '·'}</span>
                      <span className="p">{d.p.join('') || '·'}</span>
                    </div>
                  ))}
                </div>
                <div className="legend">
                  top = true tile, bottom = predicted ·
                  <span className="sw" style={{ background: '#d5f5e3' }} />
                  match
                  <span className="sw" style={{ background: '#fdebd0' }} />
                  missed
                  <span className="sw" style={{ background: '#fadbd8' }} />
                  extra
                </div>

                <h3 style={{ marginTop: 18 }}>Predicted max-score belief</h3>
                <ScoreHistogram lane={lane} />
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
