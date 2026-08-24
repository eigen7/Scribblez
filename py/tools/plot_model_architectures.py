#!/usr/bin/env python3
"""Render the architecture diagrams embedded in docs/model_architectures.md.

    py/tools/plot_model_architectures.py [--outdir docs/images] [--png-dir DIR]

One SVG per figure, named `arch_<figure>.svg`. Layout is hand-placed -- each
figure names its columns and row tops as local constants -- but box widths are
measured from the rendered text, so relabelling a box keeps it fitting its
contents. `--png-dir` additionally writes rasterized copies, which is how the
figures are eyeballed while editing them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

# Text as outlines, so a viewer without DejaVu still gets the measured layout.
matplotlib.rcParams["svg.fonttype"] = "path"
# Element ids are salted; fixing the salt keeps re-runs byte-identical, so
# regenerating an unchanged figure is a no-op in the diff.
matplotlib.rcParams["svg.hashsalt"] = "scribblez-model-architectures"

# One data unit is one point, so font sizes and layout constants share a scale.
PAD_X = 13.0
PAD_Y = 9.0
LINE_H = 17.0

INK = "#1f2328"
EDGE = "#57606a"
MUTED = "#6e7781"

# (font size, colour, family, weight) per line style.
STYLES = {
    "title": (12.0, INK, "DejaVu Sans", "bold"),
    "sub": (10.0, EDGE, "DejaVu Sans", "normal"),
    "mono": (10.0, INK, "DejaVu Sans Mono", "normal"),
}

# (fill, stroke) per box role.
PALETTE = {
    "input": ("#eef1f4", "#8c959f"),
    "trunk": ("#ddf4ff", "#0969da"),
    "op": ("#f6f8fa", "#57606a"),
    "head": ("#fff8c5", "#9a6700"),
    "out": ("#dafbe1", "#1a7f37"),
    "ghost": ("#ffffff", "#c9d1d9"),
}


def title(text: str) -> tuple[str, str]:
    return (text, "title")


def sub(text: str) -> tuple[str, str]:
    return (text, "sub")


def mono(text: str) -> tuple[str, str]:
    return (text, "mono")


@dataclass(frozen=True)
class Box:
    """A placed box, exposing the edge midpoints edges are routed between."""

    cx: float
    top: float
    w: float
    h: float

    @property
    def bottom(self) -> tuple[float, float]:
        return (self.cx, self.top + self.h)

    @property
    def head(self) -> tuple[float, float]:
        return (self.cx, self.top)

    @property
    def left(self) -> tuple[float, float]:
        return (self.cx - self.w / 2, self.top + self.h / 2)

    @property
    def right(self) -> tuple[float, float]:
        return (self.cx + self.w / 2, self.top + self.h / 2)

    def bottom_at(self, cx: float) -> tuple[float, float]:
        return (cx, self.top + self.h)


class Diagram:
    """A fixed-size canvas in point coordinates, with the origin top-left."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white")
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)  # y grows downward
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()

    def label(self, x: float, y: float, text: str, style: str = "sub", ha: str = "center"):
        size, color, family, weight = STYLES[style]
        return self.ax.text(
            x,
            y,
            text,
            size=size,
            color=color,
            family=family,
            weight=weight,
            ha=ha,
            va="center",
            zorder=4,
        )

    def _text_width(self, artist) -> float:
        bb = artist.get_window_extent(renderer=self.renderer)
        return abs(bb.transformed(self.ax.transData.inverted()).width)

    def box(self, cx: float, top: float, lines: list, kind: str = "op", min_w: float = 0.0) -> Box:
        h = 2 * PAD_Y + len(lines) * LINE_H
        drawn = [
            self.label(cx, top + PAD_Y + (i + 0.5) * LINE_H, text, style)
            for i, (text, style) in enumerate(lines)
        ]
        w = max([min_w] + [self._text_width(t) + 2 * PAD_X for t in drawn])
        face, stroke = PALETTE[kind]
        self.ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, top),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=7",
                fc=face,
                ec=stroke,
                lw=1.2,
                zorder=2,
                mutation_aspect=1,
            )
        )
        return Box(cx, top, w, h)

    def merge(self, cx: float, cy: float, symbol: str = "+", r: float = 11.0) -> Box:
        self.ax.add_patch(Circle((cx, cy), r, fc="white", ec=EDGE, lw=1.2, zorder=3))
        self.label(cx, cy - 0.5, symbol, "title")
        return Box(cx, cy - r, 2 * r, 2 * r)

    def edge(self, *points: tuple[float, float], dashed: bool = False):
        self.ax.add_patch(
            FancyArrowPatch(
                path=MplPath(list(points)),
                arrowstyle="-|>,head_length=5,head_width=2.6",
                lw=1.1,
                color=EDGE,
                linestyle="--" if dashed else "-",
                joinstyle="round",
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )

    def note(self, x: float, y: float, text: str, ha: str = "left"):
        self.ax.text(
            x, y, text, size=9.5, color=MUTED, family="DejaVu Sans", ha=ha, va="center", zorder=4
        )

    def caption(self, text: str):
        self.ax.text(
            self.width / 2,
            self.height - 14,
            text,
            size=9.5,
            color=MUTED,
            family="DejaVu Sans",
            ha="center",
            va="center",
        )

    def save(self, svg_path: Path, png_path: Path | None):
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(svg_path, format="svg", metadata={"Date": None})
        if png_path is not None:
            png_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(png_path, format="png", dpi=144)
        plt.close(self.fig)


def spatial_trunk() -> Diagram:
    """Fig 1: the shared board front end."""
    d = Diagram(760, 556)
    left, right = 215, 540

    spatial = d.box(left, 26, [title("input_spatial"), mono("(B, P_in, 15, 15)")], "input")
    scalar = d.box(right, 26, [title("input_scalar"), mono("(B, S_in)")], "input")

    stem = d.box(
        left, 118, [title("stem"), sub("conv 3×3   P_in → C"), sub("BatchNorm → ReLU")], "trunk"
    )
    proj = d.box(
        right,
        118,
        [title("scalar_proj"), sub("Linear S_in → C → ReLU"), sub("Linear C → C")],
        "trunk",
    )
    d.edge(spatial.bottom, stem.head)
    d.edge(scalar.bottom, proj.head)

    plus = d.merge(left, 240)
    d.edge(stem.bottom, plus.head)
    d.note(left + 16, 213, "x  (B, C, 15, 15)")

    tower = d.box(
        left,
        286,
        [
            title("residual tower  -  N blocks"),
            sub("block i:  i % 3 == 2  →  GlobalPoolingResBlock"),
            sub("otherwise  →  ResBlock"),
        ],
        "trunk",
    )
    d.edge(plus.bottom, tower.head)

    norm = d.box(left, 396, [title("BatchNorm → ReLU")], "trunk")
    d.edge(tower.bottom, norm.head)

    out_x = d.box(left, 466, [title("x"), mono("(B, C, 15, 15)")], "out")
    out_s = d.box(right, 466, [title("s"), mono("(B, C)")], "out")
    d.edge(norm.bottom, out_x.head)

    # The scalar projection runs down the right margin: broadcast into the board
    # features at the stem, and handed to the heads as `s`.
    d.edge((right, proj.top + proj.h), (right, out_s.top))
    d.edge((right, 240), (left + 11, 240))
    d.note(right - 18, 226, "broadcast-add over 15 × 15", ha="right")
    d.note(right - 18, 214, "(1+γ)·x + s with use_film", ha="right")
    d.note(right + 14, 400, "s  (B, C)")
    return d


def _res_block(d: Diagram, cx: float, skip_x: float):
    d.label(cx, 26, "ResBlock   (pre-activation)", "title")
    src = d.box(cx, 50, [title("x")], "input")
    b1 = d.box(cx, 112, [title("BatchNorm → ReLU"), sub("conv 3×3   C → C")], "op")
    b2 = d.box(cx, 202, [title("BatchNorm → ReLU"), sub("conv 3×3   C → C")], "op")
    plus = d.merge(cx, 305)
    out = d.box(cx, 340, [title("out")], "out")
    d.edge(src.bottom, b1.head)
    d.edge(b1.bottom, b2.head)
    d.edge(b2.bottom, plus.head)
    d.edge(plus.bottom, out.head)
    d.edge(
        src.left,
        (skip_x, src.left[1]),
        (skip_x, 305),
        (cx - 11, 305),
    )
    d.note(skip_x + 8, 200, "skip")


def _gpool_block(d: Diagram, cx: float, spatial_x: float, pool_x: float, skip_x: float):
    d.label(cx, 26, "GlobalPoolingResBlock   (KataGo-style)", "title")
    src = d.box(cx, 50, [title("x")], "input")
    b1 = d.box(cx, 112, [title("BatchNorm → ReLU"), sub("conv 3×3   C → C")], "op")
    d.edge(src.bottom, b1.head)

    spatial = d.box(spatial_x, 202, [title("channels [0 : Cs]"), sub("spatial branch")], "op")
    pool = d.box(
        pool_x,
        202,
        [
            title("channels [Cs : C]"),
            sub("mean_max_pool  →  (B, 2 Cp)"),
            sub("Linear 2 Cp → Cs"),
        ],
        "op",
    )
    d.edge(b1.bottom, (cx, 182), (spatial_x, 182), spatial.head)
    d.edge(b1.bottom, (cx, 182), (pool_x, 182), pool.head)

    plus = d.merge(spatial_x, 320)
    d.edge(spatial.bottom, plus.head)
    d.edge(pool.bottom, (pool_x, 320), (spatial_x + 11, 320))
    d.note(pool_x - 14, 306, "per-channel bias β (+ gain γ with use_film)", ha="right")

    b2 = d.box(spatial_x, 356, [title("BatchNorm → ReLU"), sub("conv 3×3   Cs → C")], "op")
    d.edge(plus.bottom, b2.head)

    plus2 = d.merge(spatial_x, 460)
    out = d.box(spatial_x, 495, [title("out")], "out")
    d.edge(b2.bottom, plus2.head)
    d.edge(plus2.bottom, out.head)
    d.edge(
        src.left,
        (skip_x, src.left[1]),
        (skip_x, 460),
        (spatial_x - 11, 460),
    )
    d.note(skip_x + 8, 300, "skip")


def tower_blocks() -> Diagram:
    """Fig 2: the two block types the tower interleaves."""
    d = Diagram(880, 580)
    _res_block(d, 150, 48)
    _gpool_block(d, 580, 490, 715, 372)
    d.ax.plot([300, 300], [40, 540], color="#d0d7de", lw=1.0, ls=(0, (4, 4)), zorder=0)
    d.caption("Cp = C // 2,   Cs = C - Cp")
    return d


def position_eval() -> Diagram:
    """Fig 3: PositionEvalModel's six heads."""
    d = Diagram(920, 700)
    x_col, s_col, v_col, mask_col = 370, 675, 520, 140

    spatial = d.box(x_col, 26, [title("input_spatial"), mono("(B, P_in, 15, 15)")], "input")
    scalar = d.box(s_col, 26, [title("input_scalar"), mono("(B, S_in)")], "input")
    trunk = d.box(v_col, 112, [title("SpatialTrunk"), sub("fig. 1")], "trunk", min_w=320)
    d.edge(spatial.bottom, (x_col, 96), (x_col, trunk.top))
    d.edge(scalar.bottom, (s_col, 96), (s_col, trunk.top))

    feat = d.box(x_col, 194, [mono("x   (B, C, 15, 15)")], "ghost")
    proj = d.box(s_col, 194, [mono("s   (B, C)")], "ghost")
    d.edge(trunk.bottom_at(x_col), feat.head)
    d.edge(trunk.bottom_at(s_col), proj.head)

    pool = d.box(x_col, 264, [title("mean_max_pool"), mono("(B, 2C)")], "op")
    d.edge(feat.bottom, pool.head)

    cat = d.merge(v_col, 356, symbol="‖")
    d.edge(pool.bottom, (x_col, 356), (v_col - 11, 356))
    d.edge(proj.bottom, (s_col, 356), (v_col + 11, 356))
    d.note(v_col + 22, 340, "concat")

    value = d.box(v_col, 388, [title("v   value summary"), mono("(B, 3C)")], "op")
    d.edge(cat.bottom, value.head)

    wld_fc = d.box(
        340, 486, [title("wld_fc"), sub("Linear 3C → 64 → ReLU"), sub("Linear 64 → 3")], "head"
    )
    mean_fc = d.box(
        565,
        486,
        [title("sd_mean_fc"), sub("Linear 3C → 256 → ReLU"), sub("Linear 256 → 1")],
        "head",
    )
    std_fc = d.box(
        790,
        486,
        [
            title("sd_std_fc"),
            sub("reads  v.detach()"),
            sub("Linear 3C → 256 → ReLU"),
            sub("Linear 256 → 1, softplus"),
        ],
        "head",
    )
    for head in (wld_fc, mean_fc, std_fc):
        d.edge(value.bottom, (v_col, 466), (head.cx, 466), head.head)

    wld_out = d.box(340, 612, [title("wld"), mono("(B, 3) logits")], "out")
    sd_out = d.box(660, 612, [title("score_diff"), mono("(B, 2) = [mean, std]")], "out")
    d.edge(wld_fc.bottom, wld_out.head)
    d.edge(mean_fc.bottom, (565, 592), (620, 592), (620, sd_out.top))
    d.edge(std_fc.bottom, (790, 592), (700, 592), (700, sd_out.top))

    mask_conv = d.box(
        mask_col,
        264,
        [
            title("mask_conv"),
            sub("conv 1×1  C → 32, no bias"),
            sub("BatchNorm → ReLU"),
            sub("conv 1×1  32 → 4"),
        ],
        "head",
    )
    d.edge(feat.bottom, (x_col, 240), (mask_col, 240), mask_conv.head)

    d.box(
        mask_col,
        400,
        [
            title("4 placement masks"),
            mono("(B, 15, 15) logits each"),
            sub("opp_next_placement"),
            sub("self_next_placement"),
            sub("opp_win_placement"),
            sub("self_win_placement"),
        ],
        "out",
    )
    d.edge(mask_conv.bottom, (mask_col, 400))
    return d


def move_encoder() -> Diagram:
    """Fig 4: MoveEncoder -- one query vector per candidate move."""
    d = Diagram(940, 700)
    sq_col, let_col, blank_col, sc_col = 175, 425, 630, 825

    squares = d.box(
        sq_col,
        26,
        [title("move_squares"), mono("(M, T)"), sub("with move_pos_id and board (P, 225, C)")],
        "input",
    )
    letters = d.box(let_col, 26, [title("move_letters"), mono("(M, T)")], "input")
    blanks = d.box(blank_col, 26, [title("move_blanks"), mono("(M, T)")], "input")
    scalars = d.box(sc_col, 26, [title("move_scalars"), mono("(M, S_mv)")], "input")

    gather = d.box(
        sq_col,
        118,
        [
            title("gather board tokens"),
            mono("board[move_pos_id, move_squares]"),
            sub("× is_play   (exchange / pass → 0)"),
        ],
        "op",
    )
    letter_emb = d.box(
        let_col, 118, [title("letter_emb"), sub("Embedding 27 → C"), sub("padding_idx = 0")], "op"
    )
    blank_emb = d.box(blank_col, 118, [title("blank_emb"), sub("Embedding 2 → C")], "op")
    scalar_mlp = d.box(
        sc_col,
        118,
        [title("scalar_mlp"), sub("Linear 3 → C → ReLU"), sub("Linear C → C")],
        "op",
    )
    for src, dst in (
        (squares, gather),
        (letters, letter_emb),
        (blanks, blank_emb),
        (scalars, scalar_mlp),
    ):
        d.edge(src.bottom, dst.head)

    plus = d.merge(let_col, 248)
    d.edge(gather.bottom, (sq_col, 248), (let_col - 11, 248))
    d.edge(letter_emb.bottom, plus.head)
    d.edge(blank_emb.bottom, (blank_col, 248), (let_col + 11, 248))

    tile_tok = d.box(let_col, 282, [title("tile_tok"), mono("(M, T, C)")], "op")
    d.edge(plus.bottom, tile_tok.head)

    pooled = d.box(
        let_col,
        374,
        [
            title("masked mean over T"),
            sub("× move_tile_mask, / real-tile count"),
            mono("tile_pool   (M, C)"),
        ],
        "op",
    )
    d.edge(tile_tok.bottom, pooled.head)

    feat = d.box(sc_col, 282, [mono("scalar_feat   (M, C)")], "ghost")
    d.edge(scalar_mlp.bottom, feat.head)

    concat = d.merge(620, 490, symbol="‖")
    d.edge(pooled.bottom, (let_col, 490), (620 - 11, 490))
    d.edge(feat.bottom, (sc_col, 490), (620 + 11, 490))
    d.note(620, 466, "concat  (M, 2C)")

    fuse = d.box(620, 524, [title("fuse"), sub("Linear 2C → C")], "op")
    d.edge(concat.bottom, fuse.head)

    out = d.box(620, 614, [title("e   move query"), mono("(M, C)")], "out")
    d.edge(fuse.bottom, out.head)
    return d


def move_set_eval() -> Diagram:
    """Fig 5: MoveSetEvalModel -- one board encode, M candidates scored."""
    d = Diagram(960, 1010)
    x_col, s_col, board_col, main_col, move_col = 250, 500, 175, 440, 730
    bypass_x = 900

    spatial = d.box(x_col, 26, [title("input_spatial"), mono("(P, P_in, 15, 15)")], "input")
    scalar = d.box(s_col, 26, [title("input_scalar"), mono("(P, S_in)")], "input")
    moves_in = d.box(
        move_col,
        26,
        [
            title("move inputs"),
            sub("letters, blanks, squares"),
            sub("tile_mask, scalars, pos_id"),
        ],
        "input",
    )
    trunk = d.box(375, 122, [title("SpatialTrunk"), sub("fig. 1")], "trunk", min_w=300)
    d.edge(spatial.bottom, (x_col, 106), (x_col, trunk.top))
    d.edge(scalar.bottom, (s_col, 106), (s_col, trunk.top))

    feat = d.box(x_col, 198, [mono("x   (P, C, 15, 15)")], "ghost")
    proj = d.box(s_col, 198, [mono("s   (P, C)")], "ghost")
    d.edge(trunk.bottom_at(x_col), feat.head)
    d.edge(trunk.bottom_at(s_col), proj.head)

    board = d.box(
        board_col,
        272,
        [
            title("board tokens"),
            sub("flatten(2) transpose + board_pos_emb"),
            mono("board   (P, 225, C)"),
        ],
        "op",
    )
    d.edge(feat.bottom, (x_col, 252), (board_col, 252), board.head)

    g = d.box(
        565,
        272,
        [title("g   position summary"), sub("mean_max_pool(x) ‖ s"), mono("(P, 3C)")],
        "op",
    )
    d.edge(feat.bottom, (x_col, 252), (540, 252), (540, g.top))
    d.edge(proj.bottom, (s_col, 252), (590, 252), (590, g.top))

    encoder = d.box(move_col, 392, [title("MoveEncoder"), sub("fig. 4"), mono("e   (M, C)")], "op")
    d.edge(moves_in.bottom, (move_col, encoder.top))
    d.edge(board.bottom, (board_col, 428), (move_col - encoder.w / 2, 428))
    d.note(board_col + 14, 414, "board token at each placed tile's square")

    queries = d.box(
        move_col,
        502,
        [title("scatter by (pos_id, rank)"), mono("queries   (P, maxK, C)")],
        "op",
    )
    d.edge(encoder.bottom, queries.head)

    attn = d.box(
        main_col,
        600,
        [
            title("MultiheadAttention   4 heads"),
            sub("Q = queries      K = V = board"),
            sub("need_weights = False"),
        ],
        "head",
    )
    d.edge(board.bottom, (board_col, 580), (main_col - 90, 580), (main_col - 90, attn.top))
    d.edge(queries.bottom, (move_col, 580), (main_col + 90, 580), (main_col + 90, attn.top))

    attended = d.box(
        main_col, 710, [title("gather by (pos_id, rank)"), mono("attended   (M, C)")], "op"
    )
    d.edge(attn.bottom, attended.head)

    concat = d.merge(main_col, 800, symbol="‖")
    d.edge(attended.bottom, concat.head)
    d.edge(g.right, (bypass_x, g.top + g.h / 2), (bypass_x, 800), (main_col + 11, 800))
    d.note(bypass_x - 14, 786, "g[move_pos_id]   (M, 3C)", ha="right")

    head = d.box(
        main_col,
        834,
        [title("head"), sub("Linear 4C → C → ReLU"), sub("Linear C → 5")],
        "head",
    )
    d.edge(concat.bottom, head.head)

    wld = d.box(280, 940, [title("wld"), mono("(M, 3) logits")], "out")
    sd = d.box(620, 940, [title("score_diff"), mono("(M, 2) = [mean, std]")], "out")
    d.edge(head.bottom, (main_col, 920), (280, 920), wld.head)
    d.edge(head.bottom, (main_col, 920), (620, 920), sd.head)
    d.note(300, 906, "out[:, 0:3]")
    d.note(640, 906, "out[:, 3], softplus(out[:, 4]) + 1e-3")
    return d


def evidence_fusion() -> Diagram:
    """Fig 6: EvidenceFusion -- an evidence set conditioning the board map."""
    d = Diagram(980, 1150)
    move_col, plane_col, sc_col, tok_col = 180, 490, 800, 490

    moves_in = d.box(
        move_col,
        26,
        [
            title("evidence move inputs"),
            sub("letters, blanks, squares,"),
            sub("tile_mask, scalars"),
            mono("(P, E, ...)"),
        ],
        "input",
    )
    planes_in = d.box(
        plane_col,
        26,
        [
            title("obs_planes"),
            sub("4 observed ‖ 4 predicted ‖ footprint"),
            mono("(P, E, 9, 15, 15)"),
        ],
        "input",
    )
    scalars_in = d.box(
        sc_col,
        26,
        [
            title("obs_scalars"),
            sub("sim value + counts ‖"),
            sub("predicted value"),
            mono("(P, E, 11)"),
        ],
        "input",
    )

    encoder = d.box(
        move_col,
        170,
        [title("MoveEncoder"), sub("fig. 4, reused -- reads the plain board"), mono("(P, E, C)")],
        "op",
    )
    conv = d.box(
        plane_col,
        170,
        [
            title("plane_conv"),
            sub("Conv 9 → d 1×1 → ReLU → Conv d → d 3×3"),
            mono("feats   (P, E, d, 15, 15)"),
        ],
        "op",
    )
    mlp = d.box(
        sc_col,
        290,
        [title("scalar_mlp"), sub("Linear 11 → C → ReLU"), sub("Linear C → C")],
        "op",
    )
    for src, dst in ((moves_in, encoder), (planes_in, conv), (scalars_in, mlp)):
        d.edge(src.bottom, dst.head)

    pool = d.box(plane_col, 290, [title("mean_max_pool"), mono("(P, E, 2d)")], "op")
    d.edge(conv.bottom, pool.head)

    concat = d.merge(tok_col, 420, symbol="‖")
    d.edge(encoder.bottom, (move_col, 420), (tok_col - 11, 420))
    d.edge(pool.bottom, concat.head)
    d.edge(mlp.bottom, (sc_col, 420), (tok_col + 11, 420))

    fuse = d.box(
        tok_col,
        454,
        [title("token_fuse"), sub("Linear 2C + 2d → C"), mono("tokens   (P, E, C)")],
        "op",
    )
    d.edge(concat.bottom, fuse.head)
    d.note(tok_col + 150, 470, "cacheable per candidate")

    self_attn = d.box(
        tok_col,
        566,
        [
            title("evidence self-attention"),
            sub("TransformerEncoderLayer, padding-masked"),
            mono("t   (P, E, C)"),
        ],
        "head",
    )
    d.edge(fuse.bottom, self_attn.head)

    board = d.box(160, 566, [title("board"), mono("(P, 225, C)"), sub("fig. 5")], "ghost")
    cross = d.box(
        420,
        690,
        [
            title("cross-attention   Q = board   K = V = t"),
            sub("weights also mix feats at each query square"),
            sub("out_proj + spatial_out, both zero-init"),
            mono("Δboard   (P, 225, C)"),
        ],
        "head",
    )
    d.edge(board.bottom, (160, 668), (330, 668), (330, cross.top))
    d.edge(self_attn.bottom, (tok_col, 668), (510, 668), (510, cross.top))
    d.edge(
        conv.bottom, (plane_col + 130, 250), (940, 250), (940, 668), (560, 668), (560, cross.top)
    )

    pooled = d.box(
        800,
        690,
        [title("masked mean over E"), mono("(P, C)"), sub("summary_out   zero-init")],
        "op",
    )
    d.edge(self_attn.bottom, (tok_col, 668), (pooled.head[0], 668), pooled.head)

    gate = d.box(
        420,
        850,
        [title("× has_evidence"), sub("an empty set passes through bit-exactly")],
        "op",
    )
    d.edge(cross.bottom, gate.head)

    board_out = d.box(
        300, 990, [title("board′"), mono("(P, 225, C)"), sub("board + Δboard")], "out"
    )
    g_out = d.box(680, 990, [title("g′"), mono("(P, 3C)"), sub("g + summary")], "out")
    d.edge(gate.bottom, (420, 962), (300, 962), board_out.head)
    d.edge(pooled.bottom, (800, 962), (680, 962), g_out.head)
    d.caption("scoring (fig. 5) reads board′ and g′ exactly as it reads board and g")
    return d


FIGURES = {
    "arch_spatial_trunk": spatial_trunk,
    "arch_tower_blocks": tower_blocks,
    "arch_position_eval": position_eval,
    "arch_move_encoder": move_encoder,
    "arch_move_set_eval": move_set_eval,
    "arch_evidence_fusion": evidence_fusion,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="docs/images", help="directory to write SVGs into")
    parser.add_argument("--png-dir", default=None, help="also write PNG copies here")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    png_dir = Path(args.png_dir) if args.png_dir else None
    for name, build in FIGURES.items():
        diagram = build()
        diagram.save(outdir / f"{name}.svg", png_dir / f"{name}.png" if png_dir else None)
        print(f"wrote {outdir / (name + '.svg')}")


if __name__ == "__main__":
    main()
