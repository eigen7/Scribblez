import { useContext, useEffect, useMemo, useState } from 'react';
import { TabActiveContext } from './TabActiveContext';
import GenerationSlider from './GenerationSlider';
import {
  AnswerRow, BoardColumn, BoardView, MoveRef, PositionHeader, PositionPicker, READING_COLOR, Unseen, pct,
  useKnownPositions,
} from './KnownPositionView';
import OppRackBelief from './OppRackBelief';
import { getJSON } from '../lib/api';
import { RackBeliefChart } from '../lib/surveyTypes';
import { useDebouncedValue } from '../lib/useDebouncedValue';

// deep_bot's Positions tab: one known position (positions/NWL23/known-positions)
// at a time under one of the tag's releases. The board before the move, the
// position's answer beside the release's picks there, and two of the release's
// models: the policy over PolicyBot's candidates and the leave inference
// model's belief about the opponent's rack, as the trainer computed them when it
// published the release (py/scribblez/deep_bot/release_positions.py). Clicking a
// move draws it on the board.

const POLICY_COLOR = '#3a86d4';
const REPLY_LABEL: Record<string, string> = { hasty: 'hasty reply', subtree: 'subtree reply' };

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

interface Policy {
  num_candidates: number;
  inferred: boolean;
  candidates: Candidate[];
}

interface Payload extends BoardView {
  release: number;
  picks: Record<string, Pick>;
  // Null for a release published before the trainer analyzed the positions.
  belief: RackBeliefChart | null;
  policy: Policy | null;
}

interface Release {
  release: number;
  rows_trained: Record<string, number>;
  picked: boolean; // whether the measure role has recorded its picks
}

// What a move is to this position: the answer, static equity's choice, a pick.
function moveTags(p: Payload, move: string): string[] {
  const tags = [];
  if (p.reference.some((r) => r.move === move)) tags.push('reference');
  if (p.static_equity_best.move === move) tags.push('static equity');
  for (const [reply, pick] of Object.entries(p.picks)) if (pick.move === move) tags.push(`pick · ${reply}`);
  return tags;
}

// Where the policy puts a move; nothing without the release's policy.
function policyText(byMove: Map<string, Candidate> | null, move: string): string {
  if (byMove == null) return '';
  const c = byMove.get(move);
  return c ? `policy #${c.policy_rank} · ${pct(c.prob)}` : 'not a policy candidate';
}

function Answers({
  p, byMove, selected, onSelect,
}: {
  p: Payload;
  byMove: Map<string, Candidate> | null;
  selected: string | null;
  onSelect: (m: string) => void;
}) {
  const replies = Object.keys(p.picks).sort();
  return (
    <div className="dbp-panel">
      <div className="dbp-panel-title">The position&apos;s answer and release {p.release}&apos;s picks</div>
      {p.reference.map((r) => (
        <AnswerRow key={`ref-${r.move}`} label="reference" move={r} color={READING_COLOR.reference}
          note={policyText(byMove, r.move)} selected={selected === r.move} onSelect={() => onSelect(r.move)} />
      ))}
      <AnswerRow label="static equity" move={p.static_equity_best} note={policyText(byMove, p.static_equity_best.move)}
        selected={selected === p.static_equity_best.move} onSelect={() => onSelect(p.static_equity_best.move)} />
      {replies.map((reply) => {
        const pick = p.picks[reply];
        return (
          <AnswerRow key={reply} label={`pick · ${REPLY_LABEL[reply] ?? reply}`} move={pick}
            color={READING_COLOR[pick.reading] ?? READING_COLOR.other} note={policyText(byMove, pick.move)}
            selected={selected === pick.move} onSelect={() => onSelect(pick.move)} />
        );
      })}
      {replies.length === 0 && <div className="muted dbp-note">No picks recorded for this release yet.</div>}
    </div>
  );
}

function PolicyTable({
  p, policy, selected, onSelect,
}: {
  p: Payload;
  policy: Policy;
  selected: string | null;
  onSelect: (m: string) => void;
}) {
  const { candidates, num_candidates: total, inferred } = policy;
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

// Every move the position's lists name, by notation, for the board preview.
function namedMoves(p: Payload): Map<string, MoveRef> {
  const out = new Map<string, MoveRef>();
  for (const m of [...(p.policy?.candidates ?? []), ...p.reference, p.static_equity_best, ...Object.values(p.picks)])
    if (!out.has(m.move)) out.set(m.move, m);
  return out;
}

function PositionView({ p, modelPending }: { p: Payload; modelPending: boolean }) {
  // Opens on the policy's choice; kept across releases while the move exists.
  const [selected, setSelected] = useState<string | null>(null);
  const moves = useMemo(() => namedMoves(p), [p]);
  const byMove = useMemo(() => (p.policy ? new Map(p.policy.candidates.map((c) => [c.move, c])) : null), [p]);
  const opening = p.policy?.candidates[0]?.move ?? null;
  const shownMove = selected != null && moves.has(selected) ? selected : opening;
  const preview = shownMove != null ? moves.get(shownMove)! : null;
  return (
    <>
      <PositionHeader p={p}>
        {modelPending && (
          <span className="dbp-pending">
            · <span className="scz-spinner" style={{ verticalAlign: 'middle' }} /> loading release…
          </span>
        )}
      </PositionHeader>
      <div className="lane-main">
        <BoardColumn p={p} preview={preview} />
        <div className="dbp-side">
          <Answers p={p} byMove={byMove} selected={shownMove} onSelect={setSelected} />
          {!modelPending && p.belief && <OppRackBelief chart={p.belief} />}
          <div style={{ width: 320 }}>
            <Unseen p={p} />
          </div>
        </div>
      </div>
      {!modelPending && p.policy && <PolicyTable p={p} policy={p.policy} selected={shownMove} onSelect={setSelected} />}
      {!modelPending && !p.policy && (
        <div className="dbp-panel dbp-note">
          Release {p.release} was published before the trainer analyzed the known positions, so it has no
          model outputs to show.
        </div>
      )}
    </>
  );
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
      <PositionPicker positions={positions} index={posIdx} onChange={setPosIdx} />
      {error && <div className="dbp-error">{error}</div>}
      {current ? (
        <PositionView key={current.id} p={current} modelPending={modelPending} />
      ) : (
        !error && <div className="muted" style={{ padding: 20 }}>Loading position…</div>
      )}
    </div>
  );
}
