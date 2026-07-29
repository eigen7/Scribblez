#!/usr/bin/env python3
"""Render the `endgame_bench --mode=endgames` margin sweep as SVG charts.

Usage:
    py/tools/plot_endgame_bench.py sweep.txt [--outdir docs/images] [--prefix endgame]

Feed it the captured stdout of the sweep; it writes <prefix>_skill_vs_margin.svg and
<prefix>_cost_vs_margin.svg, which docs/endgame_bench_results.md embeds in place of the
raw tables.
"""

import argparse
import dataclasses
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator

# Ink, surface and gridline tokens; the SVGs sit in a markdown doc that may be viewed on a
# dark page, so the surface is painted opaquely rather than inherited from the page.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"

# Budgets are an ordered ladder, so they take a single-hue ordinal ramp (light = least
# compute) rather than categorical hues. These are steps 250..700 of the blue ramp; the
# 3- and 5-series subsets this script picks were checked against the ordinal gates
# (monotone lightness, adjacent dL >= 0.06, light end >= 2:1 on the surface).
BLUE_RAMP = [
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

SKILL_TABLE_MARKER = "skill:"
COST_TABLE_MARKER = "cost:"


@dataclasses.dataclass
class Sweep:
    games: int
    captured: int
    timed_games: int
    threads: int
    margin_min: int
    margin_max: int
    margin_step: int
    margins: list[int]
    budgets: list[int]
    skill: dict[int, list[float]]
    hasty_win: list[float]
    cost: dict[int, list[float]]


def parse_header(text):
    """Pull the run's shape out of the `endgames mode (margin sweep): ...` banner."""
    match = re.search(r"^endgames mode \(margin sweep\):.*$", text, re.MULTILINE)
    if match is None:
        raise ValueError("no 'endgames mode (margin sweep):' header line found")
    line = match.group(0)
    fields = {
        "games": r"(\d+) games",
        "captured": r"(\d+) captured",
        "timed_games": r"timed games=(\d+)",
        "threads": r"threads=(\d+)",
    }
    parsed = {}
    for name, pattern in fields.items():
        field_match = re.search(pattern, line)
        if field_match is None:
            raise ValueError(f"header line has no {name!r} field: {line}")
        parsed[name] = int(field_match.group(1))
    span = re.search(r"margins (-?\d+)\.\.(-?\d+) step (\d+)", line)
    if span is None:
        raise ValueError(f"header line has no 'margins A..B step S' field: {line}")
    parsed["margin_min"] = int(span.group(1))
    parsed["margin_max"] = int(span.group(2))
    parsed["margin_step"] = int(span.group(3))
    return parsed


def _is_int(token):
    try:
        int(token)
    except ValueError:
        return False
    return True


def parse_table(lines, marker):
    """Parse one `margin x budget` table into (budgets, margins, columns, trailing).

    The budget list comes from the column header row, so adding or removing budgets in the
    benchmark needs no change here. `trailing` holds the optional extra column past the
    budgets (the skill table's `hasty win%`), or None.
    """
    starts = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if not starts:
        raise ValueError(f"no {marker!r} table found in the sweep output")
    head = lines[starts[0] + 1].split()
    if not head or head[0] != "margin":
        raise ValueError(f"{marker!r} table is not followed by a 'margin ...' column header")
    budgets = []
    for token in head[1:]:
        if not _is_int(token):
            break
        budgets.append(int(token))
    if not budgets:
        raise ValueError(f"{marker!r} column header lists no budgets: {' '.join(head)}")

    margins = []
    columns = [[] for _ in budgets]
    trailing = []
    has_trailing = None
    for line in lines[starts[0] + 2 :]:
        tokens = line.split()
        if not tokens or not _is_int(tokens[0]):
            break
        extras = len(tokens) - 1 - len(budgets)
        if extras not in (0, 1):
            raise ValueError(
                f"{marker!r} row has {len(tokens)} fields, expected "
                f"{len(budgets) + 1} or {len(budgets) + 2}: {line}"
            )
        if has_trailing is None:
            has_trailing = extras == 1
        elif has_trailing != (extras == 1):
            raise ValueError(f"{marker!r} rows disagree on the trailing column: {line}")
        margins.append(int(tokens[0]))
        for i in range(len(budgets)):
            columns[i].append(float(tokens[1 + i]))
        if has_trailing:
            trailing.append(float(tokens[-1]))
    if not margins:
        raise ValueError(f"{marker!r} table has no data rows")
    return budgets, margins, columns, (trailing if has_trailing else None)


def parse_sweep(text):
    lines = [line.rstrip() for line in text.splitlines()]
    header = parse_header(text)

    skill_budgets, skill_margins, skill_columns, hasty_win = parse_table(lines, SKILL_TABLE_MARKER)
    if hasty_win is None:
        raise ValueError("skill table is missing its trailing 'hasty win%' column")
    cost_budgets, cost_margins, cost_columns, _ = parse_table(lines, COST_TABLE_MARKER)
    if cost_budgets != skill_budgets or cost_margins != skill_margins:
        raise ValueError("skill and cost tables disagree on their margin or budget axes")

    return Sweep(
        margins=skill_margins,
        budgets=skill_budgets,
        skill=dict(zip(skill_budgets, skill_columns, strict=True)),
        hasty_win=hasty_win,
        cost=dict(zip(cost_budgets, cost_columns, strict=True)),
        **header,
    )


def budget_colors(count):
    """Evenly spaced steps of the blue ramp, lightest first."""
    if count < 1 or count > len(BLUE_RAMP):
        raise ValueError(f"cannot draw {count} budget series from a {len(BLUE_RAMP)}-step ramp")
    if count == 1:
        return [BLUE_RAMP[-1]]
    return [BLUE_RAMP[round(i * (len(BLUE_RAMP) - 1) / (count - 1))] for i in range(count)]


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "font.size": 9,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.transparent": False,
            "text.color": INK_PRIMARY,
            # Keep SVG text as text (small files, selectable labels) and keep repeated renders
            # of unchanged data byte-identical so the docs images do not churn in git.
            "svg.fonttype": "none",
            "svg.hashsalt": "endgame-bench",
        }
    )


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_RULE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, width=0.8, labelsize=8.5)
    ax.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def margin_tick_step(sweep):
    span = sweep.margins[-1] - sweep.margins[0]
    for step in (5, 10, 20, 25, 50, 100):
        if span / step <= 13:
            return step
    return span


def style_margin_axis(ax, sweep):
    ax.set_xlim(sweep.margins[0], sweep.margins[-1])
    ax.xaxis.set_major_locator(MultipleLocator(margin_tick_step(sweep)))


def draw_titles(fig, title, subtitle):
    fig.text(0.095, 0.945, title, fontsize=13, color=INK_PRIMARY, va="top")
    fig.text(0.095, 0.885, subtitle, fontsize=8.5, color=INK_SECONDARY, va="top")


def draw_budget_legend(fig, handles, count):
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=count,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SECONDARY,
        title="solver node budget (per solve)",
        title_fontsize=8.5,
        handlelength=2.2,
        columnspacing=1.8,
    )
    fig.legends[-1].get_title().set_color(INK_SECONDARY)


def subtitle_tail(sweep):
    return (
        f"{sweep.captured} endgames captured from {sweep.games} self-play games  "
        f"·  margins {sweep.margin_min}..{sweep.margin_max} step {sweep.margin_step}"
    )


def plot_skill(sweep, path):
    """Win% advantage per budget, over a subordinate panel carrying the baseline it is
    measured against. Two panels rather than two y-scales on one plot: the baseline is
    absolute win% and the series are differences, and overlaying them on invented-aligned
    axes would suggest a correlation the data does not contain."""
    fig, (ax, ax_base) = plt.subplots(
        2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.14}
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.845, bottom=0.185)

    colors = budget_colors(len(sweep.budgets))
    handles = []
    for depth, (budget, color) in enumerate(zip(sweep.budgets, colors, strict=True)):
        (line,) = ax.plot(
            sweep.margins,
            sweep.skill[budget],
            color=color,
            linewidth=1.2,
            solid_capstyle="round",
            zorder=3 + depth,
            label=f"{budget}",
        )
        handles.append(line)
    ax.axhline(0, color=AXIS_RULE, linewidth=1.0, zorder=2)

    style_axes(ax)
    ax.set_ylabel(
        "win% advantage over HastyBot\n(percentage points)", fontsize=9, color=INK_SECONDARY
    )

    ax_base.plot(sweep.margins, sweep.hasty_win, color=INK_MUTED, linewidth=1.1, zorder=3)
    style_axes(ax_base)
    style_margin_axis(ax_base, sweep)
    ax_base.set_ylim(0, 100)
    ax_base.yaxis.set_major_locator(MultipleLocator(50))
    ax_base.set_ylabel("HastyBot\nwin% (context)", fontsize=9, color=INK_SECONDARY)
    ax_base.set_xlabel(
        "start-of-endgame margin (points, first actor's seat)", fontsize=9, color=INK_SECONDARY
    )

    draw_titles(
        fig,
        "Endgame solver skill vs. start-of-endgame margin",
        "solver win% minus HastyBot win%, first actor's seat  ·  " + subtitle_tail(sweep),
    )
    draw_budget_legend(fig, handles, len(sweep.budgets))
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def format_ms(value, _pos):
    return f"{value:g}" if value >= 1 else f"{value:.1f}"


def plot_cost(sweep, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.845, bottom=0.195)

    colors = budget_colors(len(sweep.budgets))
    handles = []
    for depth, (budget, color) in enumerate(zip(sweep.budgets, colors, strict=True)):
        (line,) = ax.plot(
            sweep.margins,
            sweep.cost[budget],
            color=color,
            linewidth=1.2,
            solid_capstyle="round",
            zorder=3 + depth,
            label=f"{budget}",
        )
        handles.append(line)

    style_axes(ax)
    style_margin_axis(ax, sweep)
    ax.set_yscale("log")
    # Decade-only ticks leave two or three labels over this range; 1-2-5 subdivisions with
    # plain (non-exponent) labels keep the axis readable at millisecond magnitudes. The
    # remaining decade subdivisions stay unlabelled tick marks -- gridding them all would
    # bury the series and bloat the SVG.
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=20))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(float(x) for x in range(2, 10)), numticks=20)
    )
    ax.yaxis.set_major_formatter(FuncFormatter(format_ms))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.tick_params(which="minor", length=2, color=INK_MUTED, width=0.6)
    ax.set_ylabel("solver time per endgame (ms, log scale)", fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel(
        "start-of-endgame margin (points, first actor's seat)", fontsize=9, color=INK_SECONDARY
    )

    draw_titles(
        fig,
        "Endgame solver cost vs. start-of-endgame margin",
        f"mean over {sweep.timed_games} games timed single-threaded  ·  " + subtitle_tail(sweep),
    )
    draw_budget_legend(fig, handles, len(sweep.budgets))
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_txt", help="captured stdout of endgame_bench --mode=endgames")
    parser.add_argument("--outdir", default="docs/images", help="directory to write SVGs into")
    parser.add_argument("--prefix", default="endgame", help="filename prefix for the SVGs")
    args = parser.parse_args()

    with open(args.sweep_txt) as f:
        sweep = parse_sweep(f.read())

    os.makedirs(args.outdir, exist_ok=True)
    apply_style()
    skill_path = os.path.join(args.outdir, f"{args.prefix}_skill_vs_margin.svg")
    cost_path = os.path.join(args.outdir, f"{args.prefix}_cost_vs_margin.svg")
    plot_skill(sweep, skill_path)
    plot_cost(sweep, cost_path)
    print(skill_path)
    print(cost_path)


if __name__ == "__main__":
    main()
