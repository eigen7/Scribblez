import { useCallback, useEffect, useState } from 'react';
import Board from './components/Board';
import Rack from './components/Rack';
import UnseenTiles from './components/UnseenTiles';
import { oppRackTiles } from './lib/oppRack';
import { TileInfo } from './types';
import { SurveyData, SurveyMove, SurveyPosition, isOutsidePlay } from './lib/surveyTypes';

const NOOP = () => {};
// How long each of two selected moves holds the board before the other fades in.
const SHOW_MS = 3000;

// The list's two sections; at most one move is selected in each.
type Section = 'outside' | 'hasty';
type Selection = Record<Section, number | null>; // an index into position.moves

const SECTION_TITLE: Record<Section, string> = {
  outside: 'Best-simming plays outside the top',
  hasty: 'Hasty top moves',
};

const sectionOf = (m: SurveyMove): Section => (isOutsidePlay(m) ? 'outside' : 'hasty');
const signed = (v: number, digits = 1) => `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`;

function rackTiles(rack: string, tileScores: Record<string, number>): TileInfo[] {
  return [...rack].map((ch) =>
    ch === '?' ? { letter: '', score: 0, isBlank: true } : { letter: ch, score: tileScores[ch] ?? 0 },
  );
}

// The rack slots a move's tiles come from (a blank's slot is '?'), so the rack
// under the board shows the leave.
function usedRackIndices(rack: string, move: SurveyMove | null): Set<number> {
  const used = new Set<number>();
  for (const t of move?.tiles ?? []) {
    const want = t.isBlank ? '?' : t.letter;
    const slot = [...rack].findIndex((ch, i) => ch === want && !used.has(i));
    if (slot >= 0) used.add(slot);
  }
  return used;
}

// A position opens on its best outside play beside the top move that play was
// measured against -- the comparison the survey is about.
function openingSelection(position: SurveyPosition): Selection {
  const outside = position.moves.findIndex(isOutsidePlay);
  const versus = position.moves.findIndex(
    (m) => !isOutsidePlay(m) && m.move === position.moves[outside]?.versus,
  );
  return { outside: outside >= 0 ? outside : null, hasty: versus >= 0 ? versus : null };
}

// What each bin of the tool's histograms covers (sim/rollout_summary.h).
const scoreBinLabel = (bin: number, bins: number) =>
  bin === bins - 1 ? 'scored 100 or more' : `scored ${10 * bin}–${10 * bin + 9}`;
const marginBinLabel = (bin: number, bins: number) => {
  if (bin === 0) return 'final margin below −200';
  if (bin === bins - 1) return 'final margin +200 or more';
  const lo = -200 + 25 * (bin - 1);
  return `final margin ${signed(lo, 0)} to ${signed(lo + 24, 0)}`;
};

// A histogram as a row of bars, scaled to its tallest bin, each bar's tooltip
// saying what it covers and how many rollouts landed there. `zeroBin`, when
// given, is the bin a zero value falls in, marked so the eye finds the sign.
function Histogram({
  bins, zeroBin, binLabel,
}: { bins: number[]; zeroBin?: number; binLabel: (bin: number, bins: number) => string }) {
  const peak = Math.max(1, ...bins);
  const total = Math.max(1, bins.reduce((a, b) => a + b, 0));
  return (
    <div className="survey-hist">
      {bins.map((n, i) => (
        <div
          className={`survey-hist-slot${i === zeroBin ? ' zero' : ''}`}
          key={i}
          title={`${binLabel(i, bins.length)}: ${n} rollouts (${((100 * n) / total).toFixed(1)}%)`}
        >
          <div className="survey-hist-bar" style={{ height: `${(100 * n) / peak}%` }} />
        </div>
      ))}
    </div>
  );
}

function StatRow({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="survey-stat-row" title={hint}>
      <span className="survey-stat-label">{label}</span>
      <span className="survey-stat-value">{value}</span>
    </div>
  );
}

function NextMovePanel({ title, stats }: { title: string; stats: SurveyMove['stats']['opp_reply'] }) {
  return (
    <div className="survey-stat-group">
      <div className="survey-stat-title">{title}</div>
      <StatRow label="mean score" value={stats.score.toFixed(1)} />
      <StatRow label="bingo" value={`${stats.bingo_pct.toFixed(1)}%`} />
      <StatRow
        label="beside this move"
        value={`${stats.adjacent_pct.toFixed(1)}%`}
        hint="plays that laid a tile next to one this move placed"
      />
      <StatRow label="exchange / pass" value={`${stats.non_play_pct.toFixed(1)}%`} />
      <Histogram bins={stats.score_hist} binLabel={scoreBinLabel} />
    </div>
  );
}

function StatsPanel({ move, section }: { move: SurveyMove; section: Section }) {
  const s = move.stats;
  return (
    <div className={`survey-stats survey-tint-${section}`}>
      <div className="survey-stats-move">{move.move}</div>
      <div className="survey-stat-group">
        <StatRow label="hasty rank" value={`#${move.hasty_rank}`} />
        <StatRow label="score / equity" value={`${move.score} / ${move.equity.toFixed(1)}`} />
        <StatRow label="leave" value={move.leave || '—'} />
        {move.is_setup && <StatRow label="high-value setup" value="yes" />}
      </div>
      <div className="survey-stat-group">
        <div className="survey-stat-title">Confirming sim ({s.rollouts})</div>
        <StatRow label="win" value={`${s.win_pct.toFixed(1)}%`} />
        <StatRow label="spread" value={`${signed(s.spread)} ± ${s.spread_sd.toFixed(0)} sd`} />
        {move.sigmas !== undefined && (
          <StatRow
            label={`vs ${move.versus}`}
            value={`${signed(move.gain_pct ?? 0)} pts, ${signed(move.sigmas)}σ`}
            hint="win% against the best of the hasty top moves, paired over the same rollouts"
          />
        )}
        {/* bin 9 of the tool's 18 holds margins in [0, 25) */}
        <Histogram bins={s.delta_hist} zeroBin={9} binLabel={marginBinLabel} />
      </div>
      <NextMovePanel title="Opponent's reply" stats={s.opp_reply} />
      <NextMovePanel title="Our next move" stats={s.self_next} />
      <div className="survey-stat-group">
        <div className="survey-stat-title">Game end</div>
        <StatRow label="we play out" value={`${s.self_went_out_pct.toFixed(1)}%`} />
        <StatRow label="they play out" value={`${s.opp_went_out_pct.toFixed(1)}%`} />
        <StatRow label="stuck on their rack" value={s.opp_stranded.toFixed(1)} hint="mean tile value" />
        <StatRow label="stuck on ours" value={s.self_stranded.toFixed(1)} hint="mean tile value" />
        <StatRow label="settlement swing" value={signed(s.end_swing)} />
      </div>
    </div>
  );
}

function MoveRow({
  move, selected, played, onSelect,
}: { move: SurveyMove; selected: boolean; played: boolean; onSelect: () => void }) {
  const classes = ['survey-move', selected && 'selected', move.beats_cut && 'winner']
    .filter(Boolean)
    .join(' ');
  return (
    <button type="button" className={classes} aria-pressed={selected} onClick={onSelect}>
      <span className="survey-move-rank">#{move.hasty_rank}</span>
      <span className="survey-move-name">
        {move.move}
        {played && <span className="survey-move-played" title="the move the game played"> ●</span>}
      </span>
      <span className="survey-move-score" title="points the move scores">{move.score}</span>
      <span className="survey-move-win" title="confirming-sim win%">
        {move.stats.win_pct.toFixed(1)}%
      </span>
      <span className="survey-move-sigma">
        {move.sigmas !== undefined ? `${signed(move.sigmas)}σ` : ''}
      </span>
    </button>
  );
}

function MoveSection({
  section, position, selection, onToggle,
}: {
  section: Section;
  position: SurveyPosition;
  selection: Selection;
  onToggle: (section: Section, index: number) => void;
}) {
  return (
    <div className={`survey-move-section survey-tint-${section}`}>
      <div className="survey-moves-title">{SECTION_TITLE[section]}</div>
      {position.moves.map((m, i) =>
        sectionOf(m) !== section ? null : (
          <MoveRow
            key={m.move}
            move={m}
            selected={selection[section] === i}
            played={m.move === position.played}
            onSelect={() => onToggle(section, i)}
          />
        ),
      )}
    </div>
  );
}

function PositionHeader({
  position, index, total, onStep,
}: { position: SurveyPosition; index: number; total: number; onStep: (d: number) => void }) {
  const [mine, theirs] = [position.scores[position.mover], position.scores[1 - position.mover]];
  return (
    <div className="survey-header">
      <button type="button" onClick={() => onStep(-1)} disabled={index === 0} aria-label="previous position">
        ←
      </button>
      <div className="survey-header-text">
        <div className="survey-header-title">
          {index + 1} / {total} · {position.name}
        </div>
        <div className="survey-header-sub">
          turn {position.turn} · to move {mine}–{theirs} ({signed(mine - theirs, 0)}) · bag{' '}
          {position.bag_size} · {position.num_legal_moves} legal moves
        </div>
      </div>
      <button
        type="button"
        onClick={() => onStep(1)}
        disabled={index === total - 1}
        aria-label="next position"
      >
        →
      </button>
    </div>
  );
}

// The section whose selected move the board shows right now: the only selected
// one, or -- with a move selected in both -- each in turn, SHOW_MS apiece.
function useShownSection(selection: Selection): Section | null {
  const both = selection.outside !== null && selection.hasty !== null;
  const [turn, setTurn] = useState<Section>('outside');
  useEffect(() => {
    if (!both) return undefined;
    setTurn('outside');
    const id = setInterval(
      () => setTurn((t) => (t === 'outside' ? 'hasty' : 'outside')),
      SHOW_MS,
    );
    return () => clearInterval(id);
  }, [both, selection.outside, selection.hasty]);
  if (both) return turn;
  if (selection.outside !== null) return 'outside';
  return selection.hasty !== null ? 'hasty' : null;
}

function BoardColumn({
  data, position, selection,
}: { data: SurveyData; position: SurveyPosition; selection: Selection }) {
  const shown = useShownSection(selection);
  const move = shown === null ? null : position.moves[selection[shown]!];
  const alternating = selection.outside !== null && selection.hasty !== null;
  const classes = ['survey-board', shown && `showing-${shown}`, alternating && 'alternating']
    .filter(Boolean)
    .join(' ');
  return (
    <div className={classes}>
      <Rack
        tiles={oppRackTiles(
          position.opp_known_leave,
          position.opp_known_leave.length,
          position.opp_rack_count,
          data.tile_scores,
        )}
        usedIndices={new Set()}
        label="Opponent (green: drawn since their last move)"
        interactive={false}
        hideScoreForQuestion
      />
      {/* Keyed by the shown move, so each one's tiles mount afresh and the
          fade-in restarts even on squares both moves use. */}
      <div key={move?.move ?? 'none'}>
        <Board
          board={position.board}
          bonuses={data.bonuses}
          candidateTiles={move?.tiles ?? []}
          tileScores={data.tile_scores}
          cursorRow={null}
          cursorCol={null}
          cursorDir={null}
          interactive={false}
          onCellClick={NOOP}
          onCellDrop={NOOP}
        />
      </div>
      <Rack
        tiles={rackTiles(position.rack, data.tile_scores)}
        usedIndices={usedRackIndices(position.rack, move)}
        label="To move"
        interactive={false}
        onTileClick={NOOP}
      />
    </div>
  );
}

function UnseenPane({ data, position }: { data: SurveyData; position: SurveyPosition }) {
  return (
    <UnseenTiles
      alsoSeen={position.opp_known_leave}
      state={{
        type: 'state',
        board: position.board,
        bonuses: data.bonuses,
        rack: rackTiles(position.rack, data.tile_scores).map((t) => ({
          ...t,
          letter: t.isBlank ? '?' : t.letter,
        })),
        scores: position.scores,
        player_names: ['Player 1', 'Player 2'],
        bag_count: position.bag_size,
        opponent_rack_count: position.opp_rack_count,
        your_turn: false,
        game_over: false,
      }}
    />
  );
}

// One position's view. Mounted afresh per position (keyed by the caller), so the
// selection always opens on that position's own comparison. ↑/↓ move the
// selection within the section last clicked.
function PositionView({ data, position }: { data: SurveyData; position: SurveyPosition }) {
  const [selection, setSelection] = useState<Selection>(() => openingSelection(position));
  const [active, setActive] = useState<Section>('outside');

  const toggle = useCallback((section: Section, i: number) => {
    setActive(section);
    setSelection((s) => ({ ...s, [section]: s[section] === i ? null : i }));
  }, []);
  // Move the active section's selection to its neighbouring row.
  const nudge = useCallback(
    (d: number) => {
      const rows = position.moves.flatMap((m, i) => (sectionOf(m) === active ? [i] : []));
      if (!rows.length) return;
      setSelection((s) => {
        const at = rows.indexOf(s[active] ?? -1);
        const next = at < 0 ? 0 : Math.min(Math.max(at + d, 0), rows.length - 1);
        return { ...s, [active]: rows[next] };
      });
    },
    [position, active],
  );
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
      nudge(e.key === 'ArrowUp' ? -1 : 1);
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [nudge]);

  return (
    <div className="survey-body">
      <div className="survey-left">
        <MoveSection section="outside" position={position} selection={selection} onToggle={toggle} />
        <MoveSection section="hasty" position={position} selection={selection} onToggle={toggle} />
        <UnseenPane data={data} position={position} />
      </div>
      <BoardColumn data={data} position={position} selection={selection} />
      <div className="survey-stats-columns">
        {(['outside', 'hasty'] as const).map(
          (section) =>
            selection[section] !== null && (
              <StatsPanel key={section} section={section} move={position.moves[selection[section]!]} />
            ),
        )}
      </div>
    </div>
  );
}

// The sim-survey viewer (`?tool=survey`): step through the positions where a
// play from outside the HastyBot top moves out-simmed them. Up to one move is
// selected in each section of the list (click again to deselect); their
// confirming-sim statistics sit side by side, and the board shows the selected
// move -- both in turn, fading, when there are two. ←/→ change position.
export default function AppSurvey() {
  const [data, setData] = useState<SurveyData | null>(null);
  const [error, setError] = useState('');
  const [index, setIndex] = useState(0);

  useEffect(() => {
    fetch('/api/survey')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const total = data?.positions.length ?? 0;
  const step = useCallback(
    (d: number) => setIndex((i) => Math.min(Math.max(i + d, 0), Math.max(total - 1, 0))),
    [total],
  );
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      step(e.key === 'ArrowLeft' ? -1 : 1);
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step]);

  if (error) return <div className="survey-message">Could not load the survey: {error}</div>;
  if (!data) return <div className="survey-message">Loading…</div>;

  const position = data.positions[index];
  return (
    <div className="survey-root">
      <PositionHeader position={position} index={index} total={total} onStep={step} />
      <PositionView key={position.name} data={data} position={position} />
    </div>
  );
}
