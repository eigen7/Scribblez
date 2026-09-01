"""The trajectory pane's model side: one position-set decision point re-scored
by an evidence checkpoint at every evidence prefix.

A DecisionAnalysis pairs a loaded checkpoint with one .gcg position and its
trajectory .sobs. Its constructor runs everything the prefix does not change --
the board row and the FULL legal move list from the engine
(ffi.gcg_position_inputs), the board trunk once, the move encodings and the
plain (evidence-free) pass over every legal move -- and `conditioned(prefix)`
runs only the fusion stage and the re-score, reading the first `prefix`
trajectory candidates as evidence exactly the way the deployed loop and the
trainer do (move_set_eval.evidence.build_evidence_inputs). Prefix 0 is the
plain pass by construction (an all-masked evidence set leaves the fusion's
hard gate shut), which the pane relies on and a test asserts.

`payload` turns one prefix's outputs into what the pane renders: the
trajectory cards, the move table over every legal move ranked by conditioned
value with the argmax-gain unsimmed move marked as the loop's next sim, and
for the selected simmed candidate the sim and predicted placement planes
(the pane draws their residual). The
trainer's position-set metric (position_set_metrics) reads the same
analyses, so what it charts is what the pane shows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from scribblez.evidence.checkpoints import EvidenceCheckpoint
from scribblez.ffi import GcgPositionInputs, gcg_position_inputs
from scribblez.move_set_eval.evidence import build_evidence_inputs, observed_planes
from scribblez.move_set_eval.model import footprint_cell_marginal, win_equity
from scribblez.move_set_eval.moves import encode_moves
from scribblez.move_set_eval.targets import PLANE_NAMES
from scribblez.sim_evidence.sobs import (
    MOVE_PLAY,
    ROLE_OFF_POLICY,
    SobsPosition,
    glyph_char,
)


def _sim_stats(obs: np.ndarray) -> dict:
    """One .sobs observation record's W/D/L frequencies, win value, its
    standard error, and the delta moments, for a trajectory card."""
    n = max(int(obs["n"]), 1)
    win, draw, loss = (float(obs[k]) / n for k in ("wins", "draws", "losses"))
    value = win + 0.5 * draw
    mean = float(obs["delta_sum"]) / n
    var = max(float(obs["delta_sq_sum"]) / n - mean**2, 0.0)
    return {
        "n": int(obs["n"]),
        "win": win,
        "draw": draw,
        "loss": loss,
        "value": value,
        "value_se": float(np.sqrt(max(value * (1 - value), 0.0) / n)),
        "delta_mean": mean,
        "delta_std": float(np.sqrt(var)),
    }


def move_tiles(move: np.void) -> list[dict]:
    """A MOVE_DTYPE play's placed tiles as the web Board's candidate tiles
    ({row, col, letter, isBlank}), lane order; empty for an exchange or pass."""
    if move["type"] != MOVE_PLAY:
        return []
    start, mask, horizontal = int(move["start"]), int(move["square_mask"]), bool(move["horizontal"])
    tiles = []
    along = 0
    while mask:
        if mask & 1:
            code = int(move["glyphs"][len(tiles)])
            r, c = (start, along) if horizontal else (along, start)
            tiles.append(
                {"row": r, "col": c, "letter": glyph_char(code).upper(), "isBlank": code > 26}
            )
        mask >>= 1
        along += 1
    return tiles


def move_lane(move: np.void) -> dict | None:
    """The row or column a play runs along ({horizontal, index}), None for an
    exchange or pass."""
    if move["type"] != MOVE_PLAY:
        return None
    return {"horizontal": bool(move["horizontal"]), "index": int(move["start"])}


@dataclass
class ScoredPass:
    """A pass's per-legal-move readouts as numpy: value (N,), gain (N,) and
    the per-cell placement planes (N, 4, 15, 15) -- the footprint head's anchor
    marginal, the same per-cell view the evidence path consumes."""

    value: np.ndarray
    gain: np.ndarray
    planes: np.ndarray

    @classmethod
    def from_outputs(cls, out: dict[str, torch.Tensor]) -> ScoredPass:
        value = win_equity(torch.softmax(out["wld"].float(), dim=1))
        planes = footprint_cell_marginal(out["planes"].float())
        return cls(
            value=value.cpu().numpy(),
            gain=out["gain"].float().cpu().numpy(),
            planes=planes.cpu().numpy(),
        )


def _ranks(value: np.ndarray) -> np.ndarray:
    """0-based descending rank per entry (ties by index)."""
    order = np.argsort(-value, kind="stable")
    ranks = np.empty(len(value), dtype=np.int64)
    ranks[order] = np.arange(len(value))
    return ranks


class DecisionAnalysis:
    """One (checkpoint, position, trajectory): the prefix-independent passes
    computed once, the conditioned pass memoized per prefix."""

    def __init__(
        self,
        ckpt: EvidenceCheckpoint,
        gcg_text: str,
        sobs: SobsPosition,
        max_e: int,
        device: torch.device | str = "cpu",
    ):
        self.ckpt = ckpt
        self.sobs = sobs
        self.max_e = max_e
        self.device = torch.device(device)
        cfg = ckpt.student_cfg
        self.inputs: GcgPositionInputs = gcg_position_inputs(
            gcg_text,
            opp_leave_input=cfg["open_leaves"],
            spatial_planes=cfg["spatial_planes"],
            scalar_size=cfg["scalar_size"],
        )
        self.sim_index = self._locate_candidates()
        self._cond_cache: dict[int, ScoredPass] = {}
        self._run_plain()

    @property
    def num_legal_moves(self) -> int:
        return len(self.inputs.moves)

    def _locate_candidates(self) -> np.ndarray:
        """Each trajectory candidate's index in the legal move list (the
        generator drew from this very ranking, so byte-equal Moves exist)."""
        index = {m.tobytes(): i for i, m in enumerate(self.inputs.moves)}
        found = []
        for m in self.sobs.moves:
            i = index.get(m.tobytes())
            if i is None:
                raise ValueError(
                    "a simmed candidate is not among the position's legal moves (the sidecar "
                    "was generated under another lexicon or information condition)"
                )
            found.append(i)
        return np.asarray(found, dtype=np.int64)

    @torch.no_grad()
    def _run_plain(self):
        model = self.ckpt.model
        n = self.num_legal_moves
        spatial = torch.from_numpy(self.inputs.input_spatial).unsqueeze(0).to(self.device)
        scalar = torch.from_numpy(self.inputs.input_scalar).unsqueeze(0).to(self.device)
        enc = encode_moves(self.inputs.moves, np.full(n, self.inputs.score_diff, dtype=np.int32))
        args = tuple(
            torch.from_numpy(enc[k]).to(self.device)
            for k in ("letters", "blanks", "squares", "tile_mask", "scalars")
        )
        self._pos_id = torch.zeros(n, dtype=torch.int64, device=self.device)
        self._board, self._g = model.encode_board(spatial, scalar)
        self._e = model.encode_moves(self._board, *args, self._pos_id)
        self._plain_out = model.score_moves(self._board, self._g, self._e, self._pos_id)
        self.plain = ScoredPass.from_outputs(self._plain_out)

    def conditioned(self, prefix: int) -> ScoredPass:
        """The conditioned pass over every legal move, the first `prefix`
        trajectory candidates as evidence (memoized per prefix)."""
        if prefix not in self._cond_cache:
            self._cond_cache[prefix] = ScoredPass.from_outputs(self.conditioned_outputs(prefix))
        return self._cond_cache[prefix]

    @torch.no_grad()
    def conditioned_outputs(self, prefix: int) -> dict[str, torch.Tensor]:
        if prefix not in self.sobs.evidence_prefix_sizes():
            raise ValueError(f"prefix {prefix} is not a valid evidence prefix")
        model = self.ckpt.model
        rows = self.sim_index[:prefix]
        first_pass = {k: self._plain_out[k][rows] for k in ("wld", "score_diff", "planes")}
        evidence = build_evidence_inputs(
            self.sobs.moves[:prefix],
            self.sobs.obs[:prefix],
            self.inputs.score_diff,
            first_pass,
            max_e=self.max_e,
            dtype=self._board.dtype,
            device=self.device,
        )
        tokens, spatial_feats = model.encode_evidence(self._board, evidence)
        board_c, g_c = model.evidence_fusion(
            self._board, self._g, tokens, spatial_feats, evidence.mask
        )
        return model.score_moves(board_c, g_c, self._e, self._pos_id)

    def sim_values(self) -> np.ndarray:
        """Each trajectory candidate's sim win value, trajectory order."""
        return np.array([_sim_stats(o)["value"] for o in self.sobs.obs], dtype=np.float64)

    def observed_planes(self) -> np.ndarray:
        """(K, 4, 15, 15): the count planes normalized by rollouts."""
        return observed_planes(self.sobs.moves, self.sobs.obs)[:, :4]


def _round_planes(planes: np.ndarray) -> list:
    return np.round(planes, 4).tolist()


def _next_sim(gain: np.ndarray, sim_index: np.ndarray, prefix: int) -> int | None:
    """The loop's next acquisition at this prefix: the argmax-gain move among
    those NOT yet simmed in the prefix, or None when everything is simmed."""
    candidates = np.ones(len(gain), dtype=bool)
    candidates[sim_index[:prefix]] = False
    if not candidates.any():
        return None
    masked = np.where(candidates, gain, -np.inf)
    return int(np.argmax(masked))


def payload(
    analysis: DecisionAnalysis,
    notations: list[str],
    prefix: int,
    slot: int | None = None,
    top_n: int = 40,
) -> dict:
    """The pane's per-(position, generation, prefix) view, with the overlay
    planes of the simmed candidate at trajectory `slot` and the move table cut
    to the top `top_n` of either ranking plus every simmed candidate (the board
    bundle is the caller's, from ffi.gcg_position_board_json; `notations` is
    its `moves` list, in legal-move order)."""
    sobs = analysis.sobs
    plain, cond = analysis.plain, analysis.conditioned(prefix)
    trained = analysis.ckpt.trained
    plain_rank, cond_rank = _ranks(plain.value), _ranks(cond.value)
    sim_index = analysis.sim_index
    slot_of = {int(i): s for s, i in enumerate(sim_index)}
    stats = [_sim_stats(o) for o in sobs.obs]
    next_sim = _next_sim(cond.gain, sim_index, prefix) if trained else None

    cards = [
        {
            "slot": s,
            "index": int(i),
            "notation": notations[i],
            "score": int(sobs.moves[s]["score"]),
            "tiles": move_tiles(sobs.moves[s]),
            "lane": move_lane(sobs.moves[s]),
            "off_policy": bool(sobs.roles[s] == ROLE_OFF_POLICY),
            "in_prefix": s < prefix,
            "sim": stats[s],
            "plain_value": float(plain.value[i]),
            "cond_value": float(cond.value[i]),
            "plain_rank": int(plain_rank[i]),
            "cond_rank": int(cond_rank[i]),
        }
        for s, i in enumerate(sim_index)
    ]
    # The table's rows: the head of both rankings (so a lift or a drop across
    # the top_n boundary shows), every simmed candidate, and the next sim.
    shown = (plain_rank < top_n) | (cond_rank < top_n)
    shown[sim_index] = True
    if next_sim is not None:
        shown[next_sim] = True
    order = [i for i in np.argsort(-cond.value, kind="stable") if shown[i]]
    moves = [
        {
            "index": int(i),
            "notation": notations[i],
            "score": int(analysis.inputs.moves[i]["score"]),
            "plain_rank": int(plain_rank[i]),
            "cond_rank": int(cond_rank[i]),
            "plain_value": round(float(plain.value[i]), 4),
            "cond_value": round(float(cond.value[i]), 4),
            "gain": round(float(cond.gain[i]), 4) if trained else None,
            "slot": slot_of.get(int(i)),
            "sim_value": stats[slot_of[int(i)]]["value"] if int(i) in slot_of else None,
            "next_sim": next_sim is not None and int(i) == next_sim,
        }
        for i in order
    ]
    return {
        "prefix": prefix,
        "max_prefix": max(sobs.evidence_prefix_sizes()),
        "rollouts": sobs.rollouts,
        "num_legal_moves": analysis.num_legal_moves,
        "trained": trained,
        "score_diff": analysis.inputs.score_diff,
        "next_sim": next_sim,
        "trajectory": cards,
        "moves": moves,
        "planes": _planes_block(analysis, cond, slot),
    }


def _planes_block(analysis: DecisionAnalysis, cond: ScoredPass, slot: int | None) -> dict | None:
    """The overlay's plane pair for one simmed candidate (its trajectory slot):
    per placement head the sim count plane normalized by rollouts and the
    conditioned pass's predicted per-cell marginal at this prefix (the plain one
    at prefix 0, where the two passes coincide). None without a candidate."""
    if slot is None or not 0 <= slot < len(analysis.sim_index):
        return None
    observed = analysis.observed_planes()
    i = int(analysis.sim_index[slot])
    return {
        "slot": slot,
        "n": analysis.sobs.rollouts,
        "heads": {
            name: {
                "truth": _round_planes(observed[slot, h]),
                "pred": _round_planes(cond.planes[i, h]),
            }
            for h, name in enumerate(PLANE_NAMES)
        },
    }


def position_set_metrics(analyses: list[DecisionAnalysis]) -> dict[str, float]:
    """The trainer's position-set readout over a set's decision analyses: at
    every evidence prefix of every position, the rank (0 = best, over the
    position's simmed candidates) of the sim-best candidate under the
    conditioned value and under the plain value, averaged -- lower is
    better, and conditioned below plain is the loop learning from its sims.
    Also the mean over prefixes of whether the conditioned argmax over the
    simmed candidates is the sim-best one, against the plain baseline."""
    cond_ranks, plain_ranks, cond_hits, plain_hits = [], [], [], []
    for a in analyses:
        sims = a.sim_values()
        if len(sims) < 2:
            continue
        best = int(np.argmax(sims))
        plain_vals = a.plain.value[a.sim_index]
        plain_rank = int(_ranks(plain_vals)[best])
        for prefix in a.sobs.evidence_prefix_sizes():
            cond_vals = a.conditioned(prefix).value[a.sim_index]
            cond_ranks.append(int(_ranks(cond_vals)[best]))
            plain_ranks.append(plain_rank)
            cond_hits.append(float(int(np.argmax(cond_vals)) == best))
            plain_hits.append(float(int(np.argmax(plain_vals)) == best))
    if not cond_ranks:
        return {}
    return {
        "posset_cond_rank": float(np.mean(cond_ranks)),
        "posset_plain_rank": float(np.mean(plain_ranks)),
        "posset_cond_hit": float(np.mean(cond_hits)),
        "posset_plain_hit": float(np.mean(plain_hits)),
        "posset_rows": len(cond_ranks),
    }
