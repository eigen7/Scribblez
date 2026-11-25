#!/usr/bin/env python3
"""Render the architecture diagrams embedded in docs/model_architectures.md.

    py/tools/plot_model_architectures.py [--outdir docs/images] [--png-dir DIR]

Writes one `arch_<figure>.svg` per entry in FIGURES. Positions are placed by
hand: each figure function keeps its column x's and row tops as local
constants. Box widths are measured from the rendered text, so relabelling a box
does not require re-laying it out. Use `--png-dir` to get raster copies for
checking the figures by eye while editing them.
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
# A fixed salt for SVG element ids makes re-runs byte-identical, so regenerating
# an unchanged figure leaves no diff.
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
    """A placed box. Its properties are the edge midpoints that arrows attach to;
    `head` is the top edge, where an incoming arrow lands."""

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
            title("tower  -  N blocks"),
            sub("conv (default):  ResBlock / GlobalPoolingResBlock   (fig. 2)"),
            sub("transformer:  TransformerTower over the cells as tokens   (fig. 2b)"),
        ],
        "trunk",
    )
    d.edge(plus.bottom, tower.head)

    norm = d.box(
        left,
        396,
        [title("BatchNorm → ReLU"), sub("(transformer tower: RMSNorm → ReLU)")],
        "trunk",
    )
    d.edge(tower.bottom, norm.head)

    out_x = d.box(left, 466, [title("x"), mono("(B, C, 15, 15)")], "out")
    out_s = d.box(right, 466, [title("s"), mono("(B, C)")], "out")
    d.edge(norm.bottom, out_x.head)

    # The scalar projection runs down the right margin. It is broadcast-added into
    # the board features at the stem and also passed to the heads as `s`.
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


def transformer_tower() -> Diagram:
    """Fig 2b: the transformer tower -- one nested-bottleneck block, repeated."""
    d = Diagram(940, 660)
    cx, skip_x, pos_x = 330, 96, 730

    d.label(cx, 26, "TransformerTower   -   N NestedBottleneckTransformerBlocks", "title")
    src = d.box(
        cx,
        50,
        [
            title("tokens"),
            mono("(B, 225 + R, C)"),
            sub("the cells row-major, then R register tokens"),
        ],
        "input",
    )

    down = d.box(cx, 146, [title("RMSNorm → ReLU"), sub("Linear  C → C_mid")], "op")
    d.edge(src.bottom, down.head)

    pairs = d.box(
        cx,
        236,
        [
            title("inner_length × TransformerPair   (at C_mid)"),
            sub("x += AttentionBlock(x):  RMSNorm → q, k, v → 2D RoPE on q, k"),
            sub("→ softmax(q kᵀ / √d) v → out_proj"),
            sub("x += SwiGluBlock(x):  RMSNorm → W₂ (silu(W₁ x) · W_g x)"),
        ],
        "op",
    )
    d.edge(down.bottom, pairs.head)

    up = d.box(
        cx,
        352,
        [title("RMSNorm → ReLU"), sub("Linear  C_mid → C     zero-initialised")],
        "op",
    )
    d.edge(pairs.bottom, up.head)

    plus = d.merge(cx, 448)
    d.edge(up.bottom, plus.head)
    d.edge(src.left, (skip_x, src.left[1]), (skip_x, 448), (cx - 11, 448))
    d.note(skip_x + 8, 300, "skip")
    d.note(cx + 22, 440, "× N  (each block starts as the identity)")

    final = d.box(cx, 484, [title("RMSNorm")], "trunk")
    d.edge(plus.bottom, final.head)

    out = d.box(
        cx,
        552,
        [
            title("tokens"),
            mono("(B, 225 + R, C)"),
            sub("cells → (B, C, 15, 15) → ReLU;   registers dropped"),
        ],
        "out",
    )
    d.edge(final.bottom, out.head)

    positions = d.box(
        pos_x,
        146,
        [
            title("token positions  (x, y)"),
            sub("cells:  the grid coordinates"),
            sub("registers:  learnable, initialised off-board"),
        ],
        "input",
    )
    rope = d.box(
        pos_x,
        258,
        [
            title("2D RoPE tables per attention layer"),
            sub("angle = ω_x · x + ω_y · y"),
            sub("ω learnable per head and channel pair"),
        ],
        "op",
    )
    d.edge(positions.bottom, rope.head)
    d.edge(rope.left, (pairs.right[0], rope.left[1]))
    d.caption(
        "C_mid = transformer_mid_channels,  heads = transformer_heads,  "
        "FFN width = transformer_ffn_channels,  inner_length = 2"
    )
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
        340, 486, [title("WldHead.fc"), sub("Linear 3C → 64 → ReLU"), sub("Linear 64 → 3")], "head"
    )
    mean_fc = d.box(
        565,
        486,
        [title("ScoreDiffHead.mean_fc"), sub("Linear 3C → 256 → ReLU"), sub("Linear 256 → 1")],
        "head",
    )
    std_fc = d.box(
        790,
        486,
        [
            title("ScoreDiffHead.std_fc"),
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
            title("4 × PlacementHead"),
            sub("conv 1×1  C → 13 slots → anchored"),
            sub("pooled FC  3C → 2 (pass, extra)"),
        ],
        "head",
    )
    d.edge(feat.bottom, (x_col, 240), (mask_col, 240), mask_conv.head)

    d.box(
        mask_col,
        400,
        [
            title("4 placement heads"),
            mono("(B, 2927) footprint logits each"),
            sub("opp_next_placement"),
            sub("self_next_placement"),
            sub("opp_win_placement"),
            sub("self_win_placement"),
        ],
        "out",
    )
    d.edge(mask_conv.bottom, (mask_col, 400))
    return d


FIGURES = {
    "arch_spatial_trunk": spatial_trunk,
    "arch_tower_blocks": tower_blocks,
    "arch_transformer_tower": transformer_tower,
    "arch_position_eval": position_eval,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="docs/images", help="directory to write SVGs into")
    parser.add_argument(
        "--png-dir", default=None, help="also write PNG copies here, for previewing"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    png_dir = Path(args.png_dir) if args.png_dir else None
    for name, build in FIGURES.items():
        diagram = build()
        diagram.save(outdir / f"{name}.svg", png_dir / f"{name}.png" if png_dir else None)
        print(f"wrote {outdir / (name + '.svg')}")


if __name__ == "__main__":
    main()
