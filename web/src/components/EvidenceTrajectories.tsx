// The evidence_trajectories workload's Trajectories tab: the sequential
// evidence loop thinking on a hand-maintained position set, and the trained
// model's response to it. For one position, one checkpoint (generation 0 is
// the frozen student itself; N is the trainer's pass N-1) and one evidence
// prefix, the tab shows the trajectory the generator simmed (anchor ->
// on-policy proposals -> off-policy draws, cards dimmed beyond the prefix), the board with
// the selected simmed candidate previewed and its sim / model / residual
// placement overlay, and the move table over the legal moves re-ranked by the
// conditioned value with the loop's next acquisition (argmax proves-best gain
// among the unsimmed) marked. The conditioned pass is a correction on the
// plain student's: at prefix 0 the two coincide, and at generation 0 the
// fusion is the identity (no gain column).
import { useContext, useEffect, useMemo, useState } from 'react';
import { TabActiveContext } from './TabActiveContext';
import Board from './Board';
import GenerationSlider from './GenerationSlider';
import Rack from './Rack';
import UnseenTiles from './UnseenTiles';
import { PlacedTile, TileInfo } from '../types';
import { getJSON } from '../lib/api';
import { useDebouncedValue } from '../lib/useDebouncedValue';
import {
  buildPlacementOverlay,
  OverlayMode,
  PlacementData,
  PlacementHeadKey,
} from '../lib/placementOverlay';
import {
  HeadSelection,
  NONE_HEAD,
  PlacementHeadRadios,
  PlacementLegend,
  PlacementModeControl,
} from './PlacementOverlayControls';

const NO_USED: Set<number> = new Set();
const SIM_COLOR = '#e74c3c'; // sim outcome (red), the Positions tab's Monte-Carlo hue
const MODEL_COLOR = '#3a86d4'; // model value (blue)
const COND_COLOR = '#1e8449'; // conditioned re-score (green)

interface SimStats {
  n: number;
  win: number;
  draw: number;
  loss: number;
  value: number;
  value_se: number;
  delta_mean: number;
  delta_std: number;
}

interface Card {
  slot: number;
  index: number; // legal-move index
  notation: string;
  score: number;
  tiles: PlacedTile[];
  lane: { horizontal: boolean; index: number } | null;
  off_policy: boolean;
  in_prefix: boolean;
  sim: SimStats;
  plain_value: number;
  cond_value: number;
  plain_rank: number;
  cond_rank: number;
}

interface MoveRow {
  index: number;
  notation: string;
  score: number;
  plain_rank: number;
  cond_rank: number;
  plain_value: number;
  cond_value: number;
  gain: number | null;
  slot: number | null;
  sim_value: number | null;
  next_sim: boolean;
}

interface BoardBundle {
  board: (string | null)[][];
  bonuses: (string | null)[][];
  rack: TileInfo[];
  scores: [number, number];
  player_names: [string, string];
  bag_count: number;
  opponent_rack_count: number;
  tile_scores: Record<string, number>;
  mover: number;
  opp_leave: string;
  last_move: [number, number][];
}

// The selected candidate's overlay planes, in the Positions tab's
// PlacementData shape (truth = sim count plane / rollouts, pred = the
// conditioned pass's sigmoid at this prefix).
interface PlanesBlock extends PlacementData {
  slot: number;
}

interface Payload {
  name: string;
  set: string;
  generation: number;
  board: BoardBundle;
  prefix: number;
  max_prefix: number;
  rollouts: number;
  num_legal_moves: number;
  trained: boolean;
  score_diff: number;
  next_sim: number | null;
  trajectory: Card[];
  moves: MoveRow[];
  planes: PlanesBlock | null;
}

interface Generation {
  generation: number;
  epoch: number | null;
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const val = (v: number) => v.toFixed(3);

// One simmed candidate: its slot role, notation, sim outcome, and the plain vs
// conditioned value. Cards beyond the prefix are dimmed (they are labeled
// candidates, not evidence at this prefix); off-policy draws are never evidence.
function TrajectoryCard({
  card, selected, onSelect,
}: { card: Card; selected: boolean; onSelect: () => void }) {
  const role = card.off_policy ? 'off-policy' : card.slot === 0 ? 'anchor' : `proposal ${card.slot}`;
  return (
    <button
      type="button"
      className={`traj-card${selected ? ' selected' : ''}${card.in_prefix ? '' : ' beyond'}`}
      onClick={onSelect}
      title={`${card.notation} (+${card.score}) — sim over ${card.sim.n} rollouts: win ${pct(card.sim.win)}, draw ${pct(card.sim.draw)}, loss ${pct(card.sim.loss)}`}
    >
      <div className="traj-card-role">{role}</div>
      <div className="traj-card-move">{card.notation} <span className="muted">+{card.score}</span></div>
      <div className="traj-card-line">
        <span style={{ color: SIM_COLOR }}>sim {val(card.sim.value)}</span>
        <span className="muted"> ± {val(card.sim.value_se)}</span>
      </div>
      <div className="traj-card-line muted">Δ {card.sim.delta_mean.toFixed(1)} (σ {card.sim.delta_std.toFixed(1)})</div>
      <div className="traj-card-line">
        <span style={{ color: MODEL_COLOR }}>plain {val(card.plain_value)}</span>
        {' · '}
        <span style={{ color: COND_COLOR }}>cond {val(card.cond_value)}</span>
      </div>
    </button>
  );
}

// The legal moves re-ranked by the conditioned value at this prefix: the head
// of both rankings plus every simmed candidate (the server cuts the rest),
// with the next sim marked. Rank columns are 0-based over ALL legal moves.
function MoveTable({ moves, trained, selectedIndex, onSelect }: {
  moves: MoveRow[]; trained: boolean; selectedIndex: number | null; onSelect: (slot: number) => void;
}) {
  return (
    <table className="traj-moves">
      <thead>
        <tr>
          <th>#</th>
          <th>move</th>
          <th>score</th>
          <th title="rank under the plain (evidence-free) value">plain #</th>
          <th style={{ color: MODEL_COLOR }}>plain</th>
          <th style={{ color: COND_COLOR }}>cond</th>
          {trained && <th title="proves-best head: expected gain from simming this move next">gain</th>}
          <th style={{ color: SIM_COLOR }}>sim</th>
        </tr>
      </thead>
      <tbody>
        {moves.map((m) => {
          const simmed = m.slot != null;
          const cls = [
            simmed ? 'simmed' : '',
            m.next_sim ? 'next-sim' : '',
            m.index === selectedIndex ? 'selected' : '',
          ].join(' ').trim();
          const shift = m.plain_rank - m.cond_rank;
          return (
            <tr key={m.index} className={cls || undefined} onClick={simmed ? () => onSelect(m.slot!) : undefined}>
              <td>{m.cond_rank}</td>
              <td className="traj-notation">
                {m.notation}
                {m.next_sim && <span className="traj-next-sim" title="argmax gain among the unsimmed: the loop's next sim">next sim</span>}
              </td>
              <td>{m.score}</td>
              <td>
                {m.plain_rank}
                {shift !== 0 && (
                  <span className={shift > 0 ? 'traj-up' : 'traj-down'} title="rank change, plain → conditioned">
                    {shift > 0 ? ` ▲${shift}` : ` ▼${-shift}`}
                  </span>
                )}
              </td>
              <td>{val(m.plain_value)}</td>
              <td>{val(m.cond_value)}</td>
              {trained && <td>{m.gain == null ? '' : m.gain.toFixed(4)}</td>}
              <td>{m.sim_value == null ? '' : val(m.sim_value)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// Retry a fixed list until it loads (the data API can take ~10s to bind at
// startup: it imports torch + the engine FFI).
function usePolledList<T>(url: string | null, key: string, deps: unknown[]): T[] {
  const tabActive = useContext(TabActiveContext);
  const [items, setItems] = useState<T[]>([]);
  // Clear only when the target changes -- not on tab re-activation, so a
  // kept-alive tab shows its old list instantly while the poll refreshes it.
  useEffect(() => setItems([]), [url]);
  useEffect(() => {
    if (!url || !tabActive) return;
    let cancelled = false;
    const tryFetch = () =>
      getJSON(url)
        .then((d) => {
          if (!cancelled && d[key]?.length) setItems(d[key]);
        })
        .catch(() => {});
    tryFetch();
    const id = setInterval(tryFetch, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, tabActive, ...deps]);
  return items;
}

export default function EvidenceTrajectories({ task, tag }: { task: string; tag: string | null }) {
  const [setName, setSetName] = useState<string | null>(null);
  const [posIdx, setPosIdx] = useState(0);
  const [genIdx, setGenIdx] = useState(0);
  const [latest, setLatest] = useState(true);
  const [prefix, setPrefix] = useState<number | null>(null); // null = the largest
  const [slot, setSlot] = useState<number | null>(null); // null = last in prefix
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [headSel, setHeadSel] = useState<HeadSelection>(NONE_HEAD);
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('residual');
  const tabActive = useContext(TabActiveContext);
  const [generations, setGenerations] = useState<Generation[]>([]);

  const sets = usePolledList<string>('/api/evidence_trajectories/sets', 'sets', []);
  useEffect(() => {
    if (sets.length && (setName == null || !sets.includes(setName))) setSetName(sets[0]);
  }, [sets, setName]);
  const positions = usePolledList<string>(
    setName ? `/api/evidence_trajectories/positions?set=${encodeURIComponent(setName)}` : null,
    'positions',
    [setName],
  );

  useEffect(() => {
    if (!tag || !tabActive) return;
    const refresh = () =>
      getJSON(`/api/evidence_trajectories/generations?task=${task}&tag=${tag}`)
        .then((d: { generations: Generation[] }) =>
          setGenerations((prev) => {
            const next = d.generations;
            const same =
              prev.length === next.length &&
              prev[prev.length - 1]?.generation === next[next.length - 1]?.generation;
            return same ? prev : next;
          }),
        )
        .catch(() => {});
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [task, tag, tabActive]);

  const genCount = generations.length;
  useEffect(() => {
    if (latest) setGenIdx(Math.max(0, genCount - 1));
  }, [latest, genCount]);
  const effIdx = Math.min(genIdx, Math.max(0, genCount - 1));
  const effGen = generations[effIdx]?.generation ?? null;
  // Dragging the slider passes through every intermediate generation; only
  // fetch (and re-sim sidecars) once it settles.
  const debouncedGen = useDebouncedValue(effGen, 150);

  // A new position resets the prefix (to the largest) and the selected
  // candidate (to the prefix's last); a new prefix resets the candidate.
  useEffect(() => {
    setPrefix(null);
    setSlot(null);
  }, [setName, posIdx]);
  useEffect(() => {
    setSlot(null);
  }, [prefix]);

  useEffect(() => {
    if (!tag || !setName || positions.length === 0 || debouncedGen == null) {
      setPayload(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    const q =
      `/api/evidence_trajectories/position?task=${task}&tag=${tag}&set=${encodeURIComponent(setName)}` +
      `&position=${posIdx}&generation=${debouncedGen}` +
      (prefix == null ? '' : `&prefix=${prefix}`) +
      (slot == null ? '' : `&slot=${slot}`);
    // Raw fetch (not getJSON) so a 4xx/5xx reason reaches the UI: the first
    // request for a set sims its sidecars and can take a while or fail.
    fetch(q)
      .then(async (r) => {
        const d = await r.json();
        if (cancelled) return;
        if (!r.ok || d.error) {
          setError(d.error || `request failed (${r.status})`);
        } else {
          setError(null);
          setPayload(d);
        }
      })
      .catch(() => {
        if (!cancelled) setError('request failed');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [task, tag, setName, positions.length, posIdx, debouncedGen, prefix, slot]);

  const selectedCard = useMemo(() => {
    if (!payload) return null;
    const s = payload.planes?.slot;
    return payload.trajectory.find((c) => c.slot === s) ?? null;
  }, [payload]);

  const placementOverlay = useMemo(() => {
    if (headSel === NONE_HEAD || !payload?.planes) return null;
    return buildPlacementOverlay(payload.planes.heads, headSel, overlayMode, payload.board.board);
  }, [headSel, payload, overlayMode]);

  useEffect(() => {
    if (headSel !== NONE_HEAD && payload && !payload.planes) setHeadSel(NONE_HEAD);
  }, [payload, headSel]);

  if (!tag) return <div className="muted" style={{ padding: 20 }}>Select a tag.</div>;
  if (sets.length === 0) return <div className="muted" style={{ padding: 20 }}>No position sets under positions/NWL23/.</div>;

  const gen = generations[effIdx];
  const genLabel =
    effGen == null
      ? 'no checkpoints yet'
      : effGen === 0
        ? 'gen 0 (the frozen student)'
        : `gen ${effGen} (pass ${gen?.epoch})`;
  const effPrefix = payload ? payload.prefix : 0;
  const maxPrefix = payload ? payload.max_prefix : 0;
  const nextSim = payload?.moves.find((m) => m.next_sim) ?? null;

  return (
    <div className="lane-analysis traj-pane">
      <div className="lane-controls-row">
        <span>Model</span>
        <GenerationSlider
          count={genCount}
          valueIdx={effIdx}
          follow={latest}
          onChange={(idx, f) => {
            setGenIdx(idx);
            setLatest(f);
          }}
          label={genLabel}
        />
      </div>
      <div className="lane-controls-row">
        <label>
          Set{' '}
          <select value={setName ?? ''} onChange={(e) => { setSetName(e.target.value); setPosIdx(0); }}>
            {sets.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label>
          Position{' '}
          <button className="arrow" onClick={() => setPosIdx((i) => Math.max(0, i - 1))}>◀</button>{' '}
          <select value={posIdx} onChange={(e) => setPosIdx(Number(e.target.value))}>
            {positions.map((name, i) => <option key={name} value={i}>{name}</option>)}
          </select>{' '}
          <button className="arrow" onClick={() => setPosIdx((i) => Math.min(positions.length - 1, i + 1))}>▶</button>
        </label>
      </div>
      <div className="lane-controls-row">
        <span>Evidence prefix</span>
        <span className="gen-slider">
          <input
            type="range"
            min={0}
            max={maxPrefix}
            step={1}
            value={effPrefix}
            disabled={!payload}
            onChange={(e) => setPrefix(Number(e.target.value))}
          />
          <span className="muted gen-label">
            {payload
              ? `${effPrefix} of ${maxPrefix} proposer pick${maxPrefix === 1 ? '' : 's'} as evidence` +
                (effPrefix === 0 ? ' (plain pass)' : '')
              : ''}
          </span>
        </span>
      </div>

      {error && <div style={{ color: '#c0392b', fontSize: 13, margin: '4px 2px 10px' }}>{error}</div>}
      {!payload ? (
        <div className="muted" style={{ padding: 20 }}>
          {loading ? 'Loading position (a new set sims its sidecars first)…' : 'No position loaded.'}
        </div>
      ) : (
        <>
          <div style={{ fontSize: 13, color: '#2c3540', margin: '4px 2px 10px', opacity: loading ? 0.6 : 1 }}>
            <b>{payload.board.player_names[0]}</b> to move — score{' '}
            <b>{payload.board.scores[0]}</b>–{payload.board.scores[1]}
            {' · '}bag {payload.board.bag_count}
            {payload.board.opp_leave && <> · opponent's known leave <b>{payload.board.opp_leave}</b></>}
            {' · '}{payload.num_legal_moves} legal moves, {payload.trajectory.length} simmed × {payload.rollouts} rollouts
            {payload.trained ? (
              nextSim && <> · next sim: <b>{nextSim.notation}</b> (gain {nextSim.gain?.toFixed(4)})</>
            ) : (
              <span style={{ color: '#a05a00', marginLeft: 6 }}>· gen 0: plain student, conditioning is the identity, no gain head</span>
            )}
          </div>

          <div className="lane-main">
            <div className="lane-board-area">
              <Board
                board={payload.board.board}
                bonuses={payload.board.bonuses}
                candidateTiles={selectedCard?.tiles ?? []}
                tileScores={payload.board.tile_scores}
                cursorRow={null}
                cursorCol={null}
                cursorDir={null}
                lastMoveCells={new Set(payload.board.last_move.map(([r, c]) => `${r},${c}`))}
                highlightLane={selectedCard?.lane ?? null}
                interactive={false}
                onCellClick={() => {}}
                onCellDrop={() => {}}
                cellHalos={placementOverlay?.halos}
              />
            </div>
            <div style={{ width: 320, flexShrink: 0 }}>
              <UnseenTiles
                state={{
                  type: 'state',
                  board: payload.board.board,
                  bonuses: payload.board.bonuses,
                  rack: payload.board.rack,
                  scores: payload.board.scores,
                  player_names: payload.board.player_names,
                  bag_count: payload.board.bag_count,
                  opponent_rack_count: payload.board.opponent_rack_count,
                  your_turn: true,
                  game_over: false,
                }}
              />
              <PlacementHeadRadios placement={payload.planes} selected={headSel} onChange={setHeadSel} />
              {headSel !== NONE_HEAD && payload.planes && (
                <PlacementModeControl
                  head={payload.planes.heads[headSel as PlacementHeadKey]}
                  mode={overlayMode}
                  onChange={setOverlayMode}
                />
              )}
              {placementOverlay && <PlacementLegend mode={overlayMode} />}
              {selectedCard && (
                <div className="muted" style={{ marginTop: 10, fontSize: 12 }}>
                  Overlay for <b>{selectedCard.notation}</b>: sim planes over its {selectedCard.sim.n} rollouts vs the model's
                  planes {effPrefix === 0 ? '(plain pass)' : `conditioned on the first ${effPrefix}`}.
                </div>
              )}
            </div>
          </div>

          <div className="lane-rack">
            <Rack tiles={payload.board.rack} usedIndices={NO_USED} label={`Rack (${payload.board.player_names[0]})`} interactive={false} />
          </div>

          <div className="lane-detail" style={{ maxWidth: 'none' }}>
            <h3>Trajectory — anchor → on-policy → off-policy (click a card to preview it)</h3>
            <div className="traj-strip">
              {payload.trajectory.map((c) => (
                <TrajectoryCard key={c.slot} card={c} selected={c.slot === payload.planes?.slot} onSelect={() => setSlot(c.slot)} />
              ))}
            </div>
            <div className="legend" style={{ marginTop: 6 }}>
              <span><span className="sw" style={{ background: SIM_COLOR }} />sim value (win + ½ draw over the rollouts, ± SE)</span>
              <span><span className="sw" style={{ background: MODEL_COLOR }} />plain (evidence-free) value</span>
              <span><span className="sw" style={{ background: COND_COLOR }} />conditioned value at this prefix</span>
            </div>

            <h3 style={{ marginTop: 18 }}>Legal moves by conditioned value (top of either ranking + every simmed candidate)</h3>
            <MoveTable
              moves={payload.moves}
              trained={payload.trained}
              selectedIndex={selectedCard?.index ?? null}
              onSelect={setSlot}
            />
          </div>
        </>
      )}
    </div>
  );
}
