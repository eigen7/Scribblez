import { useContext, useEffect, useMemo, useState } from 'react';
import { TabActiveContext } from './TabActiveContext';
import GenerationSlider from './GenerationSlider';
import LeafEvalHistory from './LeafEvalHistory';
import {
  AnswerRow, BoardColumn, BoardView, MoveRef, PositionHeader, PositionPicker, READING_COLOR, Unseen, pct,
  useKnownPositions,
} from './KnownPositionView';
import { getJSON } from '../lib/api';
import { useDebouncedValue } from '../lib/useDebouncedValue';

// deep_bot_leaf_eval's Positions tab: one known position at a time under one
// generation's leaf network, as the match slot analyzed it after that
// generation's match (py/scribblez/deep_bot_leaf_eval/positions.py). LeafBot's
// value of every legal move (the leaf's win value after the
// move, its mean over opponent leaves from the leave inference model), and
// the move the matches' LeafBot plays: the best of HastyBot's top candidates.
// Clicking a move draws it on the board and charts its value over the
// generations.

const LEAF_COLOR = '#3a86d4';

interface Valued extends MoveRef {
  value: number; // the leaf's win value, 0..1
  points?: number; // what the move scores; absent from analyses written before the pick tool reported it
  equity_rank: number;
  leaf_rank: number;
}

interface Payload extends BoardView {
  generation: number;
  candidates: number;
  pick: MoveRef & { reading: string };
  num_values: number;
  values: Valued[];
  match: { score: number; ci_half_width: number; games: number; opponent: string } | null;
}

interface Generation {
  generation: number;
  rows_trained: number | null;
  score: number | null;
  ci_half_width: number | null;
}

// Where the leaf puts a move, or that it has no value (a position with a
// single legal move).
function leafText(byMove: Map<string, Valued>, move: string): string {
  const v = byMove.get(move);
  return v ? `leaf #${v.leaf_rank} · ${pct(v.value)}` : 'not valued';
}

function moveTags(p: Payload, move: string): string[] {
  const tags = [];
  if (p.reference.some((r) => r.move === move)) tags.push('reference');
  if (p.static_equity_best.move === move) tags.push('static equity');
  if (p.pick.move === move) tags.push('pick');
  return tags;
}

function Answers({
  p, byMove, selected, onSelect,
}: {
  p: Payload;
  byMove: Map<string, Valued>;
  selected: string | null;
  onSelect: (m: string) => void;
}) {
  return (
    <div className="dbp-panel">
      <div className="dbp-panel-title">The position&apos;s answer and generation {p.generation}&apos;s pick</div>
      {p.reference.map((r) => (
        <AnswerRow key={`ref-${r.move}`} label="reference" move={r} color={READING_COLOR.reference}
          note={leafText(byMove, r.move)} selected={selected === r.move} onSelect={() => onSelect(r.move)} />
      ))}
      <AnswerRow label="static equity" move={p.static_equity_best} note={leafText(byMove, p.static_equity_best.move)}
        selected={selected === p.static_equity_best.move} onSelect={() => onSelect(p.static_equity_best.move)} />
      <AnswerRow label={`pick · top ${p.candidates}`} move={p.pick}
        color={READING_COLOR[p.pick.reading] ?? READING_COLOR.other} note={leafText(byMove, p.pick.move)}
        selected={selected === p.pick.move} onSelect={() => onSelect(p.pick.move)} />
    </div>
  );
}

function ValueTable({
  p, selected, onSelect,
}: {
  p: Payload;
  selected: string | null;
  onSelect: (m: string) => void;
}) {
  const top = p.values.length ? p.values[0].value : 1;
  return (
    <div className="dbp-panel">
      <div className="dbp-panel-title">Leaf · generation {p.generation}</div>
      <div className="dbp-note">
        {p.num_values} moves, best first by the leaf&apos;s win value after the move (its mean over
        the opponent&apos;s leaves from the leave inference model). The matches&apos; LeafBot chooses among
        HastyBot&apos;s top {p.candidates} by static equity.
      </div>
      <table className="dbp-policy">
        <thead>
          <tr>
            <th className="num">#</th><th>move</th><th className="num">points</th><th className="num">leaf eval</th>
            <th /><th className="num">equity rank</th><th />
          </tr>
        </thead>
        <tbody>
          {p.values.map((v, i) => {
            const gap = i > 0 && v.leaf_rank > p.values[i - 1].leaf_rank + 1;
            return (
              <tr key={v.move} className={`${selected === v.move ? 'selected' : ''}${gap ? ' gap' : ''}`}
                onClick={() => onSelect(v.move)}>
                <td className="num">{v.leaf_rank}</td>
                <td className="mv">{v.display}</td>
                <td className="num">{v.points ?? '—'}</td>
                <td className="num">{pct(v.value)}</td>
                <td className="bar">
                  <span style={{ width: `${Math.max(1, (v.value / Math.max(top, 1e-9)) * 100)}%`, background: LEAF_COLOR }} />
                </td>
                <td className="num">{v.equity_rank > 0 ? `#${v.equity_rank}` : '—'}</td>
                <td className="tags">{moveTags(p, v.move).join(', ')}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Every move the position's lists name, by notation, for the board preview.
function namedMoves(p: Payload): Map<string, MoveRef> {
  const out = new Map<string, MoveRef>();
  for (const m of [...p.values, ...p.reference, p.static_equity_best, p.pick]) if (!out.has(m.move)) out.set(m.move, m);
  return out;
}

// Where the tab's data comes from, for the history chart's fetch.
interface Source {
  task: string;
  tag: string;
  position: number;
  newest: number | null;
}

function PositionView({ p, pending, source }: { p: Payload; pending: boolean; source: Source }) {
  // Opens on the pick; kept across generations while the move exists.
  const [selected, setSelected] = useState<string | null>(null);
  const moves = useMemo(() => namedMoves(p), [p]);
  const byMove = useMemo(() => new Map(p.values.map((v) => [v.move, v])), [p]);
  const shownMove = selected != null && moves.has(selected) ? selected : p.pick.move;
  const preview = moves.get(shownMove) ?? null;
  return (
    <>
      <PositionHeader p={p}>
        {pending && (
          <span className="dbp-pending">
            · <span className="scz-spinner" style={{ verticalAlign: 'middle' }} /> loading generation…
          </span>
        )}
      </PositionHeader>
      <div className="lane-main">
        <BoardColumn p={p} preview={preview} />
        <div className="dbp-side">
          {!pending && <Answers p={p} byMove={byMove} selected={shownMove} onSelect={setSelected} />}
          <div style={{ width: 320 }}>
            <Unseen p={p} />
          </div>
          {!pending && preview && (
            <LeafEvalHistory task={source.task} tag={source.tag} position={source.position} move={preview.move}
              display={preview.display} generation={p.generation} newest={source.newest} />
          )}
        </div>
      </div>
      {!pending && <ValueTable p={p} selected={shownMove} onSelect={setSelected} />}
    </>
  );
}

const sameGenerations = (a: Generation[], b: Generation[]) =>
  a.length === b.length && a.every((g, i) => g.generation === b[i].generation && g.score === b[i].score);

function useGenerations(task: string, tag: string | null, active: boolean): Generation[] {
  const [generations, setGenerations] = useState<Generation[]>([]);
  useEffect(() => {
    if (!tag || !active) return;
    const refresh = () =>
      getJSON(`/api/deep_bot_leaf_eval/generations?task=${task}&tag=${tag}`)
        .then((d: { generations: Generation[] }) =>
          // Keep the array while nothing changed, so a poll does not re-render.
          setGenerations((prev) => (sameGenerations(prev, d.generations) ? prev : d.generations)),
        )
        .catch(() => {});
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [task, tag, active]);
  return generations;
}

function generationLabel(g: Generation | undefined): string {
  if (!g) return '';
  const rows = g.rows_trained != null ? ` (${g.rows_trained.toLocaleString()} rows)` : '';
  const score = g.score != null ? ` · match ${g.score.toFixed(3)} ± ${(g.ci_half_width ?? 0).toFixed(3)}` : '';
  return `generation ${g.generation}${rows}${score}`;
}

export default function LeafEvalPositions({ task, tag }: { task: string; tag: string | null }) {
  const active = useContext(TabActiveContext);
  const positions = useKnownPositions(active);
  const generations = useGenerations(task, tag, active);
  const [posIdx, setPosIdx] = useState(0);
  const [genIdx, setGenIdx] = useState(0);
  const [latest, setLatest] = useState(true);
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (latest) setGenIdx(Math.max(0, generations.length - 1));
  }, [latest, generations.length]);
  const effIdx = Math.min(genIdx, Math.max(0, generations.length - 1));
  const effGen = generations[effIdx]?.generation ?? null;
  // Fetch once the slider settles; a response a newer request superseded is dropped.
  const debounced = useDebouncedValue(effGen, 150);

  useEffect(() => {
    if (!tag || positions.length === 0 || debounced == null) return;
    let cancelled = false;
    getJSON(`/api/deep_bot_leaf_eval/position?task=${task}&tag=${tag}&position=${posIdx}&generation=${debounced}`)
      .then((d) => { if (!cancelled) { setPayload(d); setError(null); } })
      .catch((e: Error) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [task, tag, positions.length, posIdx, debounced]);

  if (!tag) return <div className="muted" style={{ padding: 20 }}>Select a tag.</div>;
  if (positions.length === 0) return <div className="muted" style={{ padding: 20 }}>Loading the known positions…</div>;
  if (generations.length === 0) {
    return (
      <div className="muted" style={{ padding: 20 }}>
        No generation analyzed yet: the match slot analyzes the known positions after each match.
      </div>
    );
  }

  const posId = positions[posIdx]?.id;
  // The slider runs ahead of the payload: the board is the position's and stays,
  // but another generation's values are withheld rather than shown under this
  // generation's label.
  const current = payload != null && payload.id === posId ? payload : null;
  const pending = current != null && current.generation !== effGen;
  return (
    <div className="lane-analysis dbp">
      <div className="lane-controls-row">
        <span>Generation</span>
        <GenerationSlider
          count={generations.length}
          valueIdx={effIdx}
          follow={latest}
          onChange={(idx, f) => { setGenIdx(idx); setLatest(f); }}
          label={generationLabel(generations[effIdx])}
        />
      </div>
      <PositionPicker positions={positions} index={posIdx} onChange={setPosIdx} />
      {error && <div className="dbp-error">{error}</div>}
      {current ? (
        <PositionView key={current.id} p={current} pending={pending}
          source={{ task, tag, position: posIdx, newest: generations[generations.length - 1]?.generation ?? null }} />
      ) : (
        !error && <div className="muted" style={{ padding: 20 }}>Loading position…</div>
      )}
    </div>
  );
}
