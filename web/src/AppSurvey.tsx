import { useCallback, useEffect, useState } from 'react';
import Board from './components/Board';
import Rack from './components/Rack';
import { TileInfo } from './types';
import { SurveyData, SurveyMove, SurveyPosition, isOutsidePlay } from './lib/surveyTypes';

const NOOP = () => {};

const signed = (v: number, digits = 1) => `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`;

function rackTiles(rack: string, tileScores: Record<string, number>): TileInfo[] {
  return [...rack].map((ch) =>
    ch === '?' ? { letter: '', score: 0, isBlank: true } : { letter: ch, score: tileScores[ch] ?? 0 },
  );
}

// The rack slots a move's tiles come from (a blank's slot is '?'), so the rack
// under the board shows the leave.
function usedRackIndices(rack: string, move: SurveyMove): Set<number> {
  const used = new Set<number>();
  for (const t of move.tiles) {
    const want = t.isBlank ? '?' : t.letter;
    const slot = [...rack].findIndex((ch, i) => ch === want && !used.has(i));
    if (slot >= 0) used.add(slot);
  }
  return used;
}

// A histogram as a row of bars, scaled to its tallest bin. `zeroBin`, when
// given, is the bin a zero value falls in, marked so the eye finds the sign.
function Histogram({ bins, zeroBin, title }: { bins: number[]; zeroBin?: number; title: string }) {
  const peak = Math.max(1, ...bins);
  return (
    <div className="survey-hist" title={title}>
      {bins.map((n, i) => (
        <div className={`survey-hist-slot${i === zeroBin ? ' zero' : ''}`} key={i}>
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
        label="beside this move's tiles"
        value={`${stats.adjacent_pct.toFixed(1)}%`}
        hint="plays that laid a tile next to one this move placed"
      />
      <StatRow label="exchange / pass" value={`${stats.non_play_pct.toFixed(1)}%`} />
      <Histogram bins={stats.score_hist} title="score, by tens (last bar: 100+)" />
    </div>
  );
}

function StatsPanel({ move }: { move: SurveyMove }) {
  const s = move.stats;
  return (
    <div className="survey-stats">
      <div className="survey-stats-move">{move.move}</div>
      <div className="survey-stat-group">
        <StatRow label="hasty rank" value={`#${move.hasty_rank}`} />
        <StatRow label="score / equity" value={`${move.score} / ${move.equity.toFixed(1)}`} />
        <StatRow label="leave" value={move.leave || '—'} />
        {move.is_setup && <StatRow label="high-value setup" value="yes" />}
      </div>
      <div className="survey-stat-group">
        <div className="survey-stat-title">Confirming sim ({s.rollouts} rollouts)</div>
        <StatRow label="win" value={`${s.win_pct.toFixed(1)}%`} />
        <StatRow label="spread" value={`${signed(s.spread)} ± ${s.spread_sd.toFixed(0)} sd`} />
        {move.sigmas !== undefined && (
          <StatRow
            label={`vs ${move.versus}`}
            value={`${signed(move.gain_pct ?? 0)} win pts, ${signed(move.sigmas)}σ`}
            hint="against the best of the hasty top moves, paired over the same rollouts"
          />
        )}
        {/* bin 9 of the tool's 18 holds margins in [0, 25) */}
        <Histogram bins={s.delta_hist} zeroBin={9} title="final margin, by 25s from −200 to +200" />
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
    <button type="button" className={classes} onClick={onSelect}>
      <span className="survey-move-rank">#{move.hasty_rank}</span>
      <span className="survey-move-name">
        {move.move}
        {played && <span className="survey-move-played" title="the move the game played"> ●</span>}
      </span>
      <span className="survey-move-win">{move.stats.win_pct.toFixed(1)}%</span>
      <span className="survey-move-sigma">
        {move.sigmas !== undefined ? `${signed(move.sigmas)}σ` : ''}
      </span>
    </button>
  );
}

function MoveListPanel({
  position, selected, onSelect,
}: { position: SurveyPosition; selected: number; onSelect: (i: number) => void }) {
  const firstTop = position.moves.findIndex((m) => !isOutsidePlay(m));
  return (
    <div className="survey-moves">
      {position.moves.map((m, i) => (
        <div key={m.move}>
          {i === 0 && <div className="survey-moves-title">Best-simming plays outside the top</div>}
          {i === firstTop && <div className="survey-moves-title">Hasty top moves</div>}
          <MoveRow
            move={m}
            selected={i === selected}
            played={m.move === position.played}
            onSelect={() => onSelect(i)}
          />
        </div>
      ))}
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
          {position.opp_known_leave && ` · opponent kept ${position.opp_known_leave}`}
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

// The sim-survey viewer (`?tool=survey`): step through the positions where a
// play from outside the HastyBot top moves out-simmed them, pick a move to see
// it on the board, and read its confirming-sim statistics. ←/→ change position,
// ↑/↓ change move.
export default function AppSurvey() {
  const [data, setData] = useState<SurveyData | null>(null);
  const [error, setError] = useState('');
  const [index, setIndex] = useState(0);
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    fetch('/api/survey')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const total = data?.positions.length ?? 0;
  const moveCount = data?.positions[index]?.moves.length ?? 0;
  const step = useCallback(
    (d: number) => {
      setIndex((i) => Math.min(Math.max(i + d, 0), Math.max(total - 1, 0)));
      setSelected(0);
    },
    [total],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') step(-1);
      else if (e.key === 'ArrowRight') step(1);
      else if (e.key === 'ArrowUp') setSelected((s) => Math.max(s - 1, 0));
      else if (e.key === 'ArrowDown') setSelected((s) => Math.min(s + 1, moveCount - 1));
      else return;
      e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step, moveCount]);

  if (error) return <div className="survey-message">Could not load the survey: {error}</div>;
  if (!data) return <div className="survey-message">Loading…</div>;

  const position = data.positions[index];
  const move = position.moves[selected];
  return (
    <div className="survey-root">
      <PositionHeader position={position} index={index} total={total} onStep={step} />
      <div className="survey-body">
        <MoveListPanel position={position} selected={selected} onSelect={setSelected} />
        <div className="survey-board">
          <Board
            board={position.board}
            bonuses={data.bonuses}
            candidateTiles={move.tiles}
            tileScores={data.tile_scores}
            cursorRow={null}
            cursorCol={null}
            cursorDir={null}
            interactive={false}
            onCellClick={NOOP}
            onCellDrop={NOOP}
          />
          <Rack
            tiles={rackTiles(position.rack, data.tile_scores)}
            usedIndices={usedRackIndices(position.rack, move)}
            label=""
            interactive={false}
            onTileClick={NOOP}
          />
        </div>
        <StatsPanel move={move} />
      </div>
    </div>
  );
}
