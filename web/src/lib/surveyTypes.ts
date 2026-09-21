// The sim-survey viewer's data, as py/scribblez/sim_survey_viewer.py builds it
// and py/scripts/sim_survey_viewer.py serves it at /api/survey.
import { PlacedTile } from '../types';

// One side's next move after the candidate, over the confirming sim's rollouts.
export interface NextMoveStats {
  score: number; // mean
  bingo_pct: number;
  // The commonest spot the bingos went down at (first placed tile, GCG position
  // style) and the share of all rollouts that bingoed there; null without bingos.
  bingo_spot: { at: string; pct: number } | null;
  adjacent_pct: number; // laid a tile beside one the candidate placed
  score_hist: number[]; // by tens; the last bin is 100+
}

export interface SurveyMoveStats {
  rollouts: number;
  win_pct: number;
  spread: number; // mean final margin, mover's view
  spread_sd: number;
  delta_hist: number[]; // final margin by 25s over [-200, 200), open-ended bins at both ends
  end_swing: number; // mean end-of-game rack settlement, mover's view
  opp_stranded: number; // mean tile value left on the opponent's rack
  self_stranded: number;
  self_passed_pct: number; // rollouts in which we passed at least once
  opp_passed_pct: number;
  self_went_out_pct: number;
  opp_went_out_pct: number;
  opp_reply: NextMoveStats;
  self_next: NextMoveStats;
}

export interface SurveyMove {
  move: string; // GCG notation
  display: string; // the same with played-through tiles spelled out: "A4 (mO)u(N)T"
  hasty_rank: number; // 1-based
  equity: number;
  score: number;
  leave: string;
  is_setup: boolean;
  tiles: PlacedTile[];
  stats: SurveyMoveStats;
  // Only on a play from outside the top moves: how it fared against the best of
  // them (`versus`) in the confirming sim.
  gain_pct?: number;
  sigmas?: number;
  beats_cut?: boolean;
  versus?: string;
}

export interface SurveyPosition {
  name: string;
  gcg: string;
  turn: number; // 1-based, as neural_rank_tool --turn takes it
  mover: number;
  rack: string;
  opp_known_leave: string;
  opp_rack_count: number;
  scores: [number, number];
  bag_size: number;
  solved_endgames: boolean; // the confirming sim's rollouts solved their endgames
  num_legal_moves: number;
  played: string;
  board: (string | null)[][];
  moves: SurveyMove[]; // the outside plays, best first, then the top moves by hasty rank
}

export interface SurveyData {
  survey: { cut: number; rollouts: number; confirm_rollouts: number; recipe: string };
  bonuses: (string | null)[][];
  tile_scores: Record<string, number>;
  positions: SurveyPosition[];
}

export const isOutsidePlay = (m: SurveyMove) => m.sigmas !== undefined;
