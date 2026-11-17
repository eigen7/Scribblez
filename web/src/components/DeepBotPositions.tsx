import { useContext, useEffect, useMemo, useState } from 'react';
import { TabActiveContext } from './TabActiveContext';
import Board from './Board';
import GenerationSlider from './GenerationSlider';
import OppRackBelief from './OppRackBelief';
import Rack from './Rack';
import UnseenTiles from './UnseenTiles';
import { PlacedTile, TileInfo } from '../types';
import { getJSON } from '../lib/api';
import { RackBeliefChart } from '../lib/surveyTypes';
import { useDebouncedValue } from '../lib/useDebouncedValue';

// deep_bot's Positions tab: one known position (positions/NWL23/known-positions)
// at a time under one of the tag's releases. The board before the move, the
// position's answer beside the release's picks there, and two of the release's
// models: the policy over PolicyBot's candidates and the leave inference
// model's belief about the opponent's rack (py/scribblez/deep_bot/position_view.py).
// Clicking a move draws it on the board.

const NOOP = () => {};
const DIM = 15;
// How a pick reads against the position's answer (deep_bot/known_positions.py):
// the reference move, static equity's move, any other.
const READING_COLOR: Record<string, string> = { reference: '#2ca02c', equity: '#888888', other: '#e67e22' };
const PREVIEW_TINT = 'rgba(58, 134, 212, 0.55)';
const POLICY_COLOR = '#3a86d4';
const REPLY_LABEL: Record<string, string> = { hasty: 'hasty reply', subtree: 'subtree reply' };

interface MoveRef {
  move: string; // GCG notation, the key the lists match on
  display: string;
  tiles: PlacedTile[];
}

interface Pick extends MoveRef {
  reading: string; // reference, equity or other
  seconds: number;
}

interface Candidate extends MoveRef {
  prob: number;
  win_pct: number;
  win_sd_pct: number;
  equity_rank: number; // 0 for a pass outside the ranking
  policy_rank: number;
}

interface Payload {
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
  release: number;
  picks: Record<string, Pick>;
  belief: RackBeliefChart;
  policy: { num_candidates: number; inferred: boolean; candidates: Candidate[] };
}

interface Release {
  release: number;
  rows_trained: Record<string, number>;
  picked: boolean; // whether the measure role has recorded its picks
}

interface KnownPosition {
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
// below 1%, so the policy's long tail still reads apart from zero.
const pct = (p: number) => `${p >= 0.01 ? (100 * p).toFixed(1) : (100 * p).toPrecision(2)}%`;

// What a move is to this position: the answer, static equity's choice, a pick.
function moveTags(p: Payload, move: string): string[] {
  const tags = [];
  if (p.reference.some((r) => r.move === move)) tags.push('reference');
  if (p.static_equity_best.move === move) tags.push('static equity');
  for (const [reply, pick] of Object.entries(p.picks)) if (pick.move === move) tags.push(`pick · ${reply}`);
  return tags;
}

function AnswerRow({
  label, move, color, candidate, selected, onSelect,
}: {
  label: string;
  move: MoveRef;
  color?: string;
  candidate: Candidate | undefined;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button className={`dbp-answer${selected ? ' selected' : ''}`} onClick={onSelect}>
      <span className="dbp-answer-label">{label}</span>
      <span className="dbp-answer-move" style={color ? { color } : undefined}>{move.display}</span>
      <span className="dbp-answer-policy">
        {candidate ? `policy #${candidate.policy_rank} · ${pct(candidate.prob)}` : 'not a policy candidate'}
      </span>
    </button>
  );
}

function Answers({
  p, byMove, selected, onSelect,
}: {
  p: Payload;
  byMove: Map<string, Candidate>;
  selected: string | null;
  onSelect: (m: string) => void;
}) {
  const replies = Object.keys(p.picks).sort();
  return (
    <div className="dbp-panel">
      <div className="dbp-panel-title">The position&apos;s answer and release {p.release}&apos;s picks</div>
      {p.reference.map((r) => (
        <AnswerRow key={`ref-${r.move}`} label="reference" move={r} color={READING_COLOR.reference}
          candidate={byMove.get(r.move)} selected={selected === r.move} onSelect={() => onSelect(r.move)} />
      ))}
      <AnswerRow label="static equity" move={p.static_equity_best} candidate={byMove.get(p.static_equity_best.move)}
        selected={selected === p.static_equity_best.move} onSelect={() => onSelect(p.static_equity_best.move)} />
      {replies.map((reply) => {
        const pick = p.picks[reply];
        return (
          <AnswerRow key={reply} label={`pick · ${REPLY_LABEL[reply] ?? reply}`} move={pick}
            color={READING_COLOR[pick.reading] ?? READING_COLOR.other} candidate={byMove.get(pick.move)}
            selected={selected === pick.move} onSelect={() => onSelect(pick.move)} />
        );
      })}
      {replies.length === 0 && <div className="muted dbp-note">No picks recorded for this release yet.</div>}
    </div>
  );
}

function PolicyTable({
  p, selected, onSelect,
}: {
  p: Payload;
  selected: string | null;
  onSelect: (m: string) => void;
}) {
  const { candidates, num_candidates: total, inferred } = p.policy;
  const top = candidates.length ? candidates[0].prob : 1;
  return (
    <div className="dbp-panel">
      <div className="dbp-panel-title">Policy · release {p.release}</div>
      <div className="dbp-note">
        {total} candidates (PolicyBot&apos;s pool), likeliest first; the opponent-rack input is{' '}
        {inferred ? 'the belief on the right' : 'a uniform rack (this policy trained on uniform beliefs)'}.
        Value: the value head&apos;s win probability after the move, ± one rollout&apos;s spread.
      </div>
      <table className="dbp-policy">
        <thead>
          <tr>
            <th>#</th><th>move</th><th colSpan={2}>policy</th><th>value</th><th>equity</th><th />
          </tr>
        </thead>
        <tbody>
          {candidates.map((c, i) => {
            const gap = i > 0 && c.policy_rank > candidates[i - 1].policy_rank + 1;
            return (
              <tr key={c.move} className={`${selected === c.move ? 'selected' : ''}${gap ? ' gap' : ''}`}
                onClick={() => onSelect(c.move)}>
                <td className="num">{c.policy_rank}</td>
                <td className="mv">{c.display}</td>
                <td className="num">{pct(c.prob)}</td>
                <td className="bar">
                  <span style={{ width: `${Math.max(1, (c.prob / top) * 100)}%`, background: POLICY_COLOR }} />
                </td>
                <td className="num">{c.win_pct.toFixed(1)}% <span className="muted">± {c.win_sd_pct.toFixed(1)}</span></td>
                <td className="num">{c.equity_rank > 0 ? `#${c.equity_rank}` : '—'}</td>
                <td className="tags">{moveTags(p, c.move).join(', ')}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function BoardColumn({ p, preview }: { p: Payload; preview: MoveRef | null }) {
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

function Unseen({ p }: { p: Payload }) {
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

// Every move the position's lists name, by notation, for the board preview.
function namedMoves(p: Payload): Map<string, MoveRef> {
  const out = new Map<string, MoveRef>();
  for (const m of [...p.policy.candidates, ...p.reference, p.static_equity_best, ...Object.values(p.picks)])
    if (!out.has(m.move)) out.set(m.move, m);
  return out;
}

function PositionView({ p, modelPending }: { p: Payload; modelPending: boolean }) {
  // Opens on the policy's choice; kept across releases while the move exists.
  const [selected, setSelected] = useState<string | null>(null);
  const moves = useMemo(() => namedMoves(p), [p]);
  const byMove = useMemo(() => new Map(p.policy.candidates.map((c) => [c.move, c])), [p]);
  const shownMove = selected != null && moves.has(selected) ? selected : (p.policy.candidates[0]?.move ?? null);
  const preview = shownMove != null ? moves.get(shownMove)! : null;
  return (
    <>
      <div className="dbp-header">
        <b>{p.title}</b> · turn {p.turn} · Player {p.mover + 1} to move · {p.scores[0]}–{p.scores[1]} · bag {p.bag_size}
        {modelPending && (
          <span className="dbp-pending">
            · <span className="scz-spinner" style={{ verticalAlign: 'middle' }} /> loading release…
          </span>
        )}
      </div>
      <div className="lane-main">
        <BoardColumn p={p} preview={preview} />
        <div className="dbp-side">
          <Answers p={p} byMove={byMove} selected={shownMove} onSelect={setSelected} />
          {!modelPending && <OppRackBelief chart={p.belief} />}
          <div style={{ width: 320 }}>
            <Unseen p={p} />
          </div>
        </div>
      </div>
      {!modelPending && <PolicyTable p={p} selected={shownMove} onSelect={setSelected} />}
    </>
  );
}

function useKnownPositions(active: boolean): KnownPosition[] {
  const [positions, setPositions] = useState<KnownPosition[]>([]);
  // Retry until they load: at startup the data API can take ~10s to bind.
  useEffect(() => {
    if (positions.length > 0 || !active) return;
    let cancelled = false;
    const tryFetch = () =>
      getJSON('/api/deep_bot/positions')
        .then((d) => { if (!cancelled && d.positions.length) setPositions(d.positions); })
        .catch(() => {});
    tryFetch();
    const id = setInterval(tryFetch, 3000);
    return () => { cancelled = true; clearInterval(id); };
  }, [positions.length, active]);
  return positions;
}

const sameReleases = (a: Release[], b: Release[]) =>
  a.length === b.length && a.every((r, i) => r.release === b[i].release && r.picked === b[i].picked);

function useReleases(task: string, tag: string | null, active: boolean): Release[] {
  const [releases, setReleases] = useState<Release[]>([]);
  useEffect(() => {
    if (!tag || !active) return;
    const refresh = () =>
      getJSON(`/api/deep_bot/releases?task=${task}&tag=${tag}`)
        .then((d: { releases: Release[] }) =>
          // Keep the array while nothing changed, so a poll does not re-render.
          setReleases((prev) => (sameReleases(prev, d.releases) ? prev : d.releases)),
        )
        .catch(() => {});
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [task, tag, active]);
  return releases;
}

export default function DeepBotPositions({ task, tag }: { task: string; tag: string | null }) {
  const active = useContext(TabActiveContext);
  const positions = useKnownPositions(active);
  const releases = useReleases(task, tag, active);
  const [posIdx, setPosIdx] = useState(0);
  const [relIdx, setRelIdx] = useState(0);
  const [latest, setLatest] = useState(true);
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (latest) setRelIdx(Math.max(0, releases.length - 1));
  }, [latest, releases.length]);
  const effIdx = Math.min(relIdx, Math.max(0, releases.length - 1));
  const effRelease = releases[effIdx]?.release ?? null;
  // The release's picks arrive after it does; their arrival refetches.
  const picked = releases[effIdx]?.picked ?? false;
  // Fetch once the slider settles; a response a newer request superseded is dropped.
  const debounced = useDebouncedValue(effRelease, 150);

  useEffect(() => {
    if (!tag || positions.length === 0 || debounced == null) return;
    let cancelled = false;
    getJSON(`/api/deep_bot/position?task=${task}&tag=${tag}&position=${posIdx}&release=${debounced}`)
      .then((d) => { if (!cancelled) { setPayload(d); setError(null); } })
      .catch((e: Error) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [task, tag, positions.length, posIdx, debounced, picked]);

  if (!tag) return <div className="muted" style={{ padding: 20 }}>Select a tag.</div>;
  if (positions.length === 0) return <div className="muted" style={{ padding: 20 }}>Loading the known positions…</div>;
  if (releases.length === 0) return <div className="muted" style={{ padding: 20 }}>No release yet.</div>;

  const rows = releases[effIdx]?.rows_trained;
  const label = `release ${effRelease}${rows ? ` (policy ${rows.policy.toLocaleString()} rows)` : ''}`;
  const posId = positions[posIdx]?.id;
  // The slider runs ahead of the payload: the board is the position's and stays,
  // but another release's model outputs are withheld rather than shown under
  // this release's label.
  const current = payload != null && payload.id === posId ? payload : null;
  const modelPending = current != null && current.release !== effRelease;
  return (
    <div className="lane-analysis dbp">
      <div className="lane-controls-row">
        <span>Release</span>
        <GenerationSlider
          count={releases.length}
          valueIdx={effIdx}
          follow={latest}
          onChange={(idx, f) => { setRelIdx(idx); setLatest(f); }}
          label={label}
        />
      </div>
      <div className="lane-controls-row">
        <label>
          Position{' '}
          <button className="arrow" onClick={() => setPosIdx((i) => Math.max(0, i - 1))}>◀</button>{' '}
          <select value={posIdx} onChange={(e) => setPosIdx(Number(e.target.value))}>
            {positions.map((p, i) => (
              <option key={p.id} value={i}>{p.id} — {p.title}</option>
            ))}
          </select>{' '}
          <button className="arrow" onClick={() => setPosIdx((i) => Math.min(positions.length - 1, i + 1))}>▶</button>
        </label>
      </div>
      {error && <div className="dbp-error">{error}</div>}
      {current ? (
        <PositionView key={current.id} p={current} modelPending={modelPending} />
      ) : (
        !error && <div className="muted" style={{ padding: 20 }}>Loading position…</div>
      )}
    </div>
  );
}
