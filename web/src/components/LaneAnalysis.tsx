import { useEffect, useState } from 'react';
import Board from './Board';
import Rack from './Rack';
import { TileInfo } from '../types';
import { getJSON } from '../lib/api';

// The score head has 100 bins; the top bin is the catch-all for scores >= 99.
const TOP_SCORE_BIN = 99;
const NO_USED: Set<number> = new Set();

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
  rack: TileInfo[];
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

// One lane's status: a SQUARE marks the move sub-task, a CIRCLE the score sub-task
// (filled = the model got it right, outline = wrong). Only lanes with a legal move
// show marks.
function LaneStatusCell({ lane, selected, onClick }: { lane: Lane; selected: boolean; onClick: () => void }) {
  if (!lane.has_move) return <div className="lane-cell empty" onClick={onClick} />;
  return (
    <div className={`lane-cell${selected ? ' selected' : ''}`} onClick={onClick}>
      <span className={`lane-mark move${lane.move_correct ? ' ok' : ''}`} title="best move (square)" />
      <span className={`lane-mark score${lane.score_correct ? ' ok' : ''}`} title="best score (circle)" />
    </div>
  );
}

function IconLegend() {
  return (
    <div className="icon-legend">
      <div className="il-row">
        <span className="lane-mark move ok" />
        <span className="lane-mark move" /> best <b>move</b>
      </div>
      <div className="il-row">
        <span className="lane-mark score ok" />
        <span className="lane-mark score" /> best <b>score</b>
      </div>
      <div className="il-note">
        square = move, circle = score.
        <br />
        filled = model correct, outline = wrong.
      </div>
    </div>
  );
}

function ScoreHistogram({ lane }: { lane: Lane }) {
  const pmf = lane.pred_score_pmf;
  if (!pmf) return <div className="muted" style={{ fontStyle: 'italic' }}>No prediction yet.</div>;
  const max = Math.max(...pmf, 1e-9);
  const truth = Math.min(lane.max_score, TOP_SCORE_BIN);
  const argmax = lane.pred_score_bin ?? 0;
  return (
    <div className="score-hist-block">
      {/* The predicted argmax value, above its column (and above every bar). */}
      <div className="hist-top">
        <span className="argmax-num" style={{ left: `${argmax + 0.5}%` }}>{argmax}</span>
      </div>
      <div className="score-hist">
        {Array.from({ length: 9 }, (_, k) => (
          <div key={`g${k}`} className="hgroup-line" style={{ left: `${(k + 1) * 10}%` }} />
        ))}
        {pmf.map((v, i) => (
          <div
            key={i}
            className={`hbar${i === truth ? ' truth' : ''}`}
            style={{ height: `${(v / max) * 100}%` }}
            title={`score ${i}: ${(v * 100).toFixed(1)}%`}
          />
        ))}
      </div>
      {/* Bucket-of-10 labels under each group. */}
      <div className="hist-axis">
        {Array.from({ length: 10 }, (_, g) => (
          <div key={g} className="hgl">{`${g * 10 + 1}-${g * 10 + 10}`}</div>
        ))}
      </div>
      <div className="legend">
        <span className="sw" style={{ background: '#e67e22' }} /> true score <b>{truth}</b>
        <span className="sw" style={{ background: '#5dade2' }} /> predicted argmax <b>{argmax}</b>
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
  const [selected, setSelected] = useState<{ r: number; c: number } | null>(null);
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

  useEffect(() => {
    if (!tag) return;
    const refresh = () =>
      getJSON(`/api/lane/generations?task=${task}&tag=${tag}`)
        .then((d) => setGenerations(d.generations))
        .catch(() => {});
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [task, tag]);

  const genCount = generations.length;
  const effIdx = latest ? Math.max(0, genCount - 1) : Math.min(genIdx, Math.max(0, genCount - 1));
  const effGen = generations[effIdx]?.generation ?? null;

  // Fetch the merged board + ground-truth + prediction payload.
  useEffect(() => {
    if (!tag || positions.length === 0) {
      setPayload(null);
      return;
    }
    const g = effGen == null ? 'latest' : String(effGen);
    getJSON(`/api/lane/position?task=${task}&tag=${tag}&position=${posIdx}&generation=${g}`)
      .then(setPayload)
      .catch(() => setPayload(null));
  }, [task, tag, positions.length, posIdx, effGen]);

  // On a NEW position, default the selected lane to the globally highest-scoring
  // one (the lane holding the position's best move). Keyed on the position name,
  // so changing only the generation doesn't override a lane the user picked.
  useEffect(() => {
    if (!payload) return;
    let best = { h: true, i: -1, score: -1 };
    payload.lanes.rows.forEach((l, i) => {
      if (l.has_move && l.max_score > best.score) best = { h: true, i, score: l.max_score };
    });
    payload.lanes.cols.forEach((l, i) => {
      if (l.has_move && l.max_score > best.score) best = { h: false, i, score: l.max_score };
    });
    if (best.i >= 0) {
      setHorizontal(best.h);
      setLaneIdx(best.i);
    }
    setSelected(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [payload?.name]);

  if (!tag) return <div className="muted" style={{ padding: 20 }}>Select a tag.</div>;
  if (positions.length === 0) {
    return <div className="muted" style={{ padding: 20 }}>No lane-analysis dataset available.</div>;
  }

  const axis = payload ? (horizontal ? payload.lanes.rows : payload.lanes.cols) : [];
  const lane: Lane | undefined = axis[laneIdx];

  // Click a board square: select it (and show its lane in the current orientation).
  // Clicking the already-selected square flips orientation (horizontal <-> vertical).
  const clickSquare = (r: number, c: number) => {
    if (selected && selected.r === r && selected.c === c) {
      const nh = !horizontal;
      setHorizontal(nh);
      setLaneIdx(nh ? r : c);
    } else {
      setSelected({ r, c });
      setLaneIdx(horizontal ? r : c);
    }
  };
  // Orientation buttons keep the selected square's lane.
  const setOrient = (h: boolean) => {
    setHorizontal(h);
    if (selected) setLaneIdx(h ? selected.r : selected.c);
  };
  // Selecting a lane from a status strip targets the whole lane (no single square).
  const selectStrip = (h: boolean, i: number) => {
    setHorizontal(h);
    setLaneIdx(i);
    setSelected(null);
  };

  const genLabel =
    effGen == null
      ? 'no checkpoints yet'
      : `gen ${effGen} (${generations[effIdx]?.positions.toLocaleString()} positions)`;

  return (
    <div className="lane-analysis">
      {/* Controls: model on one row, position + orientation on the next. */}
      <div className="lane-controls-row">
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
          <span className="muted">{genLabel}</span>
        </label>
        <label>
          <input type="checkbox" checked={latest} onChange={(e) => setLatest(e.target.checked)} /> Latest
          (follow newest checkpoint)
        </label>
      </div>
      <div className="lane-controls-row">
        <label>
          Position{' '}
          <button className="arrow" onClick={() => setPosIdx((i) => Math.max(0, i - 1))}>◀</button>{' '}
          <select value={posIdx} onChange={(e) => setPosIdx(Number(e.target.value))}>
            {positions.map((name, i) => (
              <option key={name} value={i}>{name}</option>
            ))}
          </select>{' '}
          <button className="arrow" onClick={() => setPosIdx((i) => Math.min(positions.length - 1, i + 1))}>▶</button>
        </label>
        <span className="seg">
          <button className={horizontal ? 'active' : ''} onClick={() => setOrient(true)}>Horizontal</button>
          <button className={!horizontal ? 'active' : ''} onClick={() => setOrient(false)}>Vertical</button>
        </span>
      </div>

      {!payload ? (
        <div className="muted" style={{ padding: 20 }}>Loading position…</div>
      ) : (
        <>
          <div className="lane-main">
            <div className="lane-board-area">
              <Board
                board={payload.board}
                bonuses={payload.bonuses}
                candidateTiles={[]}
                tileScores={payload.tile_scores}
                cursorRow={selected?.r ?? null}
                cursorCol={selected?.c ?? null}
                cursorDir={null}
                interactive
                onCellClick={clickSquare}
                onCellDrop={() => {}}
                highlightLane={{ horizontal, index: laneIdx }}
              />
              <div className="lane-row-strip">
                {payload.lanes.rows.map((l, r) => (
                  <LaneStatusCell key={r} lane={l} selected={horizontal && laneIdx === r}
                    onClick={() => selectStrip(true, r)} />
                ))}
              </div>
              <div className="lane-col-strip">
                {payload.lanes.cols.map((l, c) => (
                  <LaneStatusCell key={c} lane={l} selected={!horizontal && laneIdx === c}
                    onClick={() => selectStrip(false, c)} />
                ))}
              </div>
            </div>
            <IconLegend />
          </div>

          {/* On-move rack (the player the model is predicting for). */}
          <div className="lane-rack">
            <Rack
              tiles={payload.rack}
              usedIndices={NO_USED}
              label={`On-move rack (Player ${payload.on_move + 1})`}
              interactive={false}
            />
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
                      <b>{m.word}</b> @ {String.fromCharCode(65 + m.col)}{m.row + 1}
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
                  top = true tile, bottom = predicted
                  <span className="sw" style={{ background: '#d5f5e3' }} /> match
                  <span className="sw" style={{ background: '#fdebd0' }} /> missed
                  <span className="sw" style={{ background: '#fadbd8' }} /> extra
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
