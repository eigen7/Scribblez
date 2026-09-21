import { useCallback, useEffect, useState } from 'react';
import Board from './components/Board';
import Rack from './components/Rack';
import UnseenTiles from './components/UnseenTiles';
import { oppRackTiles } from './lib/oppRack';
import { TileInfo } from './types';
import { SurveyData, SurveyMove, SurveyPosition, isOutsidePlay } from './lib/surveyTypes';

const NOOP = () => {};
const DIM = 15;
// Board tints of the two sections' previewed moves (index.css carries the same
// two colors as --survey-outside / --survey-hasty).
const SECTION_TINT: Record<Section, string> = {
  outside: 'rgba(230, 126, 34, 0.55)',
  hasty: 'rgba(142, 110, 220, 0.6)',
};
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

// A position opens on its best outside play beside HastyBot's own first choice.
function openingSelection(position: SurveyPosition): Selection {
  const outside = position.moves.findIndex(isOutsidePlay);
  const top = position.moves.findIndex((m) => !isOutsidePlay(m) && m.hasty_rank === 1);
  return { outside: outside >= 0 ? outside : null, hasty: top >= 0 ? top : null };
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

// A histogram drawn sideways: one thin row per bin, low values at the top, the
// bar's length its share of the rollouts. Both stats columns scale to the same
// `peakShare`, so with the columns side by side a bin sits at the same height in
// each and the two distributions compare by eye. Each row's tooltip says what it
// covers and how many rollouts landed there; `zeroBin` is marked so the eye
// finds the sign.
function Histogram({
  bins, zeroBin, binLabel, peakShare,
}: {
  bins: number[];
  zeroBin?: number;
  binLabel: (bin: number, bins: number) => string;
  peakShare: number;
}) {
  const total = Math.max(1, bins.reduce((a, b) => a + b, 0));
  return (
    <div className="survey-hist">
      {bins.map((n, i) => (
        <div
          className={`survey-hist-slot${i === zeroBin ? ' zero' : ''}`}
          key={i}
          title={`${binLabel(i, bins.length)}: ${n} rollouts (${((100 * n) / total).toFixed(1)}%)`}
        >
          <div
            className="survey-hist-bar"
            style={{ width: `${(100 * n) / total / Math.max(peakShare, 1e-9)}%` }}
          />
        </div>
      ))}
    </div>
  );
}

// The largest single-bin share of each histogram across the shown moves: the
// common scale their bars are drawn to.
interface HistPeaks {
  margin: number;
  oppReply: number;
  selfNext: number;
}

function peakShare(bins: number[]): number {
  return Math.max(...bins) / Math.max(1, bins.reduce((a, b) => a + b, 0));
}

function histPeaks(moves: SurveyMove[]): HistPeaks {
  const over = (pick: (m: SurveyMove) => number[]) => Math.max(0, ...moves.map((m) => peakShare(pick(m))));
  return {
    margin: over((m) => m.stats.delta_hist),
    oppReply: over((m) => m.stats.opp_reply.score_hist),
    selfNext: over((m) => m.stats.self_next.score_hist),
  };
}

function StatRow({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="survey-stat-row" title={hint}>
      <span className="survey-stat-label">{label}</span>
      <span className="survey-stat-value">{value}</span>
    </div>
  );
}

function NextMovePanel({
  title, stats, peak,
}: { title: string; stats: SurveyMove['stats']['opp_reply']; peak: number }) {
  return (
    <div className="survey-stat-group">
      <div className="survey-stat-title">{title}</div>
      <StatRow label="mean score" value={stats.score.toFixed(1)} />
      <StatRow label="bingo" value={`${stats.bingo_pct.toFixed(1)}%`} />
      {stats.bingo_spot ? (
        <StatRow
          label={`bingo@${stats.bingo_spot.at}`}
          value={`${stats.bingo_spot.pct.toFixed(1)}%`}
          hint="the commonest spot the bingos went down at (first tile placed; row first = across), as a share of all rollouts"
        />
      ) : (
        <StatRow label={'\u00a0'} value="" />
      )}
      <StatRow
        label="beside this move"
        value={`${stats.adjacent_pct.toFixed(1)}%`}
        hint="plays that laid a tile next to one this move placed"
      />
      <Histogram bins={stats.score_hist} binLabel={scoreBinLabel} peakShare={peak} />
    </div>
  );
}

// Every panel has the same rows in the same order -- a row that does not apply
// to a move is kept, blank -- so two panels side by side line up row for row.
function StatsPanel({
  move, section, peaks,
}: { move: SurveyMove; section: Section; peaks: HistPeaks }) {
  const s = move.stats;
  return (
    <div className={`survey-stats survey-tint-${section}`}>
      <div className="survey-stats-move">{move.display}</div>
      <div className="survey-stat-group">
        <StatRow label="hasty rank" value={`#${move.hasty_rank}`} />
        <StatRow label="score / equity" value={`${move.score} / ${move.equity.toFixed(1)}`} />
        <StatRow label="leave" value={move.leave || '—'} />
      </div>
      <div className="survey-stat-group">
        <div className="survey-stat-title">Confirming sim ({s.rollouts})</div>
        <StatRow label="win" value={`${s.win_pct.toFixed(1)}%`} />
        <StatRow label="spread" value={`${signed(s.spread)} ± ${s.spread_sd.toFixed(0)} sd`} />
        {move.sigmas !== undefined ? (
          <StatRow
            label={`vs ${move.versus}`}
            value={`${signed(move.gain_pct ?? 0)} pts, ${signed(move.sigmas)}σ`}
            hint="win% against the best of the hasty top moves, paired over the same rollouts"
          />
        ) : (
          <StatRow label={'\u00a0'} value="" />
        )}
        {/* bin 9 of the tool's 18 holds margins in [0, 25) */}
        <Histogram
          bins={s.delta_hist}
          zeroBin={9}
          binLabel={marginBinLabel}
          peakShare={peaks.margin}
        />
      </div>
      <NextMovePanel title="Opponent's reply" stats={s.opp_reply} peak={peaks.oppReply} />
      <NextMovePanel title="Our next move" stats={s.self_next} peak={peaks.selfNext} />
      <div className="survey-stat-group">
        <div className="survey-stat-title">Game end</div>
        <StatRow label="we play out" value={`${s.self_went_out_pct.toFixed(1)}%`} />
        <StatRow label="they play out" value={`${s.opp_went_out_pct.toFixed(1)}%`} />
        <StatRow label="we pass" value={`${s.self_passed_pct.toFixed(1)}%`} hint="rollouts in which we passed at least once" />
        <StatRow label="they pass" value={`${s.opp_passed_pct.toFixed(1)}%`} hint="rollouts in which they passed at least once" />
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
        {move.display}
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
          {position.bag_size} · {position.num_legal_moves} legal moves ·{' '}
          {position.solved_endgames ? 'endgames solved' : 'greedy endgames'}
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

const squaresOf = (m: SurveyMove) => new Set(m.tiles.map((t) => `${t.row},${t.col}`));

function overlap(a: SurveyMove, b: SurveyMove): boolean {
  const mine = squaresOf(a);
  return b.tiles.some((t) => mine.has(`${t.row},${t.col}`));
}

// The sections whose selected moves the board shows right now. Two moves that
// share no square are simply shown together; two that collide cannot be, so
// they take the board in turn, SHOW_MS apiece.
function useShownSections(position: SurveyPosition, selection: Selection): Section[] {
  const selected = (['outside', 'hasty'] as const).filter((s) => selection[s] !== null);
  const colliding =
    selected.length === 2 &&
    overlap(position.moves[selection.outside!], position.moves[selection.hasty!]);
  const [turn, setTurn] = useState<Section>('outside');
  useEffect(() => {
    if (!colliding) return undefined;
    setTurn('outside');
    const id = setInterval(() => setTurn((t) => (t === 'outside' ? 'hasty' : 'outside')), SHOW_MS);
    return () => clearInterval(id);
  }, [colliding, selection.outside, selection.hasty]);
  return colliding ? [turn] : selected;
}

// The per-cell tint grid <Board> draws: each shown move's squares in its
// section's color.
function moveTints(position: SurveyPosition, selection: Selection, shown: Section[]) {
  const grid: ({ color: string } | null)[][] = Array.from({ length: DIM }, () =>
    Array.from({ length: DIM }, () => null),
  );
  for (const section of shown)
    for (const t of position.moves[selection[section]!].tiles)
      grid[t.row][t.col] = { color: SECTION_TINT[section] };
  return grid;
}

function BoardColumn({
  data, position, selection,
}: { data: SurveyData; position: SurveyPosition; selection: Selection }) {
  const shown = useShownSections(position, selection);
  const moves = shown.map((section) => position.moves[selection[section]!]);
  const alternating = shown.length === 1 && selection.outside !== null && selection.hasty !== null;
  const [mine, theirs] = [position.scores[position.mover], position.scores[1 - position.mover]];
  return (
    <div className={`survey-board${alternating ? ' alternating' : ''}`}>
      <Rack
        tiles={oppRackTiles(
          position.opp_known_leave,
          position.opp_known_leave.length,
          position.opp_rack_count,
          data.tile_scores,
        )}
        usedIndices={new Set()}
        label={`Opponent · ${theirs} points (green: drawn since their last move)`}
        interactive={false}
        hideScoreForQuestion
      />
      {/* Keyed by what is shown, so an alternating move's tiles mount afresh and
          the fade-in restarts even on the squares both moves use. */}
      <div key={moves.map((m) => m.move).join('|')}>
        <Board
          board={position.board}
          bonuses={data.bonuses}
          candidateTiles={moves.flatMap((m) => m.tiles)}
          tileScores={data.tile_scores}
          cursorRow={null}
          cursorCol={null}
          cursorDir={null}
          interactive={false}
          onCellClick={NOOP}
          onCellDrop={NOOP}
          cellHighlights={moveTints(position, selection, shown)}
        />
      </div>
      {/* With one move on the board the rack shows its leave; with two, whole. */}
      <Rack
        tiles={rackTiles(position.rack, data.tile_scores)}
        usedIndices={usedRackIndices(position.rack, moves.length === 1 ? moves[0] : null)}
        label={`To move · ${mine} points`}
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

  const selectedSections = (['outside', 'hasty'] as const).filter((s) => selection[s] !== null);
  const peaks = histPeaks(selectedSections.map((s) => position.moves[selection[s]!]));
  return (
    <div className="survey-body">
      <div className="survey-left">
        <MoveSection section="outside" position={position} selection={selection} onToggle={toggle} />
        <MoveSection section="hasty" position={position} selection={selection} onToggle={toggle} />
        <UnseenPane data={data} position={position} />
      </div>
      <BoardColumn data={data} position={position} selection={selection} />
      <div className="survey-stats-columns">
        {selectedSections.map((section) => (
          <StatsPanel
            key={section}
            section={section}
            move={position.moves[selection[section]!]}
            peaks={peaks}
          />
        ))}
      </div>
    </div>
  );
}

// The sim-survey viewer (`?tool=survey`): step through the positions where a
// play from outside the HastyBot top moves out-simmed them. Up to one move is
// selected in each section of the list (click again to deselect); their
// confirming-sim statistics sit side by side, row for row, and the board shows
// the selected moves -- together, or in turn, fading, when they share a square.
// ←/→ change position.
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
