#!/usr/bin/env python3
"""Compare HastyBot equity vs. the value model's score_diff_mean as predictors
of the realized final score differential.

Input is the CSV produced by the `value_vs_equity` C++ tool (one row per sampled
held-out position):

    pov,pre_move_diff,bag_size,equity,score_diff_mean,final_diff

Usage:
    value_vs_equity <model.onnx> <mount/<tag>/test> > rows.csv   # C++ tool
    python -m scripts.value_vs_equity rows.csv                   # this script

Both estimate the same quantity -- the eventual differential -- so we ask which
one explains more of its variance. The current score margin (`pre_move_diff`) is
included as a control in every model, so each estimator is credited only for the
*move-quality* signal it adds beyond "who is already ahead":

    base : final_diff ~ 1 + pre_move_diff
    H    : final_diff ~ 1 + pre_move_diff + equity            (HastyBot)
    N    : final_diff ~ 1 + pre_move_diff + score_diff_mean   (the model)
    NH   : final_diff ~ 1 + pre_move_diff + equity + score_diff_mean

Read it as:
  * R2(N) > R2(H)  -> the model is the better value estimate; the loss at large k
    is distribution/selection (the PI loop is the fix).
  * R2(H) >= R2(N) -> the model is a *worse* predictor than equity; fix the model
    (more data / training) before any loop.
  * In NH, a significant `equity` coefficient *after* the model means the model is
    missing signal HastyBot has (most likely the leave term); a significant
    `score_diff_mean` coefficient means the model adds signal equity lacks
    (board-positional value).
"""

import argparse
import sys

import numpy as np


def ols(x_cols, y):
    """Ordinary least squares with an intercept.

    Returns (beta, r2, t_values) where beta/t_values include the intercept first.
    """
    n = len(y)
    X = np.column_stack([np.ones(n)] + x_cols)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    p = X.shape[1]
    dof = max(n - p, 1)
    sigma2 = ss_res / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.where(se > 0, beta / se, np.nan)
    return beta, r2, t_values


def compare_two(data_a, data_b, name_a, name_b) -> int:
    """Paired head-to-head: do A's predictions beat B's on the SAME positions?

    The two CSVs must be row-aligned (same test dir, same order), so `equity` and
    `final_diff` are identical between them. Comparing per-position removes the
    position-level variance that swamps the difference between two separate R^2
    numbers, so a small but real improvement (e.g. one PI iteration) shows up.
    """
    if data_a.size != data_b.size:
        print(f"row-count mismatch: {data_a.size} vs {data_b.size}", file=sys.stderr)
        return 1
    y = data_a["final_diff"].astype(float)
    m = data_a["pre_move_diff"].astype(float)
    if not (np.allclose(y, data_b["final_diff"].astype(float))
            and np.allclose(data_a["equity"].astype(float), data_b["equity"].astype(float))):
        print("CSVs are not row-aligned (equity/final_diff differ) -- regenerate both with the "
              "same data dir and --max, in the same order.", file=sys.stderr)
        return 1

    s_a = data_a["score_diff_mean"].astype(float)
    s_b = data_b["score_diff_mean"].astype(float)
    _, r2_a, _ = ols([m, s_a], y)
    _, r2_b, _ = ols([m, s_b], y)

    # Paired error comparison: positive d means A predicts closer than B on a row.
    d = (s_b - y) ** 2 - (s_a - y) ** 2
    mean_d = float(d.mean())
    t = mean_d / (float(d.std(ddof=1)) / np.sqrt(d.size)) if d.std() > 0 else float("nan")

    # Joint fit: which model's prediction survives once the other is present?
    beta, r2_joint, tvals = ols([m, s_a, s_b], y)

    print(f"n positions:                 {data_a.size}")
    print(f"R^2 (control = pre_move_diff):")
    print(f"  {name_a:24s} {r2_a:.4f}")
    print(f"  {name_b:24s} {r2_b:.4f}")
    print()
    print(f"Paired MSE improvement of {name_a} over {name_b}:")
    print(f"  mean reduction = {mean_d:+.3f} pts^2  (paired t={t:+.1f}; |t|>~2 ~ significant)")
    print()
    print("Joint fit  final_diff ~ pre_move_diff + both models  (t-stat):")
    for nm, b, tv in zip(["intercept", "pre_move_diff", name_a, name_b], beta, tvals):
        print(f"  {nm:24s} {b:+10.4f}  (t={tv:+.1f})")
    print()
    if t > 2:
        print(f"VERDICT: {name_a} predicts outcomes significantly better than {name_b} "
              f"(paired t={t:+.1f}). The retraining made a real, if small, improvement.")
    elif t < -2:
        print(f"VERDICT: {name_a} is significantly WORSE than {name_b} (paired t={t:+.1f}).")
    else:
        print(f"VERDICT: no significant difference between {name_a} and {name_b} "
              f"(paired t={t:+.1f}) -- the delta is within noise.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", help="CSV from value_vs_equity (default: stdin).")
    parser.add_argument("--vs", default="", help="Second CSV: paired head-to-head of the two "
                        "models' predictions on the same (row-aligned) positions.")
    args = parser.parse_args()

    src = args.csv if args.csv else sys.stdin
    data = np.genfromtxt(src, delimiter=",", names=True)
    n = data.size
    if n == 0:
        print("no rows", file=sys.stderr)
        return 1

    if args.vs:
        other = np.genfromtxt(args.vs, delimiter=",", names=True)
        return compare_two(data, other, args.csv or "A", args.vs)

    m = data["pre_move_diff"].astype(float)
    e = data["equity"].astype(float)
    s = data["score_diff_mean"].astype(float)
    y = data["final_diff"].astype(float)

    _, r2_base, _ = ols([m], y)
    _, r2_h, _ = ols([m, e], y)
    _, r2_n, _ = ols([m, s], y)
    beta_nh, r2_nh, t_nh = ols([m, e, s], y)

    # Standalone calibration of the model's prediction against reality.
    beta_cal, r2_cal, _ = ols([s], y)
    rmse_s = float(np.sqrt(((s - y) ** 2).mean()))
    rmse_e = float(np.sqrt(((e - y) ** 2).mean()))  # equity is not on the y scale; informational
    corr_es = float(np.corrcoef(e, s)[0, 1])

    print(f"n positions:                 {n}")
    print(f"corr(equity, score_diff):    {corr_es:+.4f}")
    print()
    print("R^2 (control = pre_move_diff in every model):")
    print(f"  base  (margin only):       {r2_base:.4f}")
    print(f"  H     (+ equity):          {r2_h:.4f}   dR2 = {r2_h - r2_base:+.4f}")
    print(f"  N     (+ score_diff_mean): {r2_n:.4f}   dR2 = {r2_n - r2_base:+.4f}")
    print(f"  NH    (+ both):            {r2_nh:.4f}")
    print()
    print("NH coefficients (t-stat; |t|>~2 ~ significant):")
    names = ["intercept", "pre_move_diff", "equity", "score_diff_mean"]
    for name, b, t in zip(names, beta_nh, t_nh):
        print(f"  {name:16s} {b:+10.4f}  (t={t:+.1f})")
    print()
    print("Model calibration vs. realized final_diff:")
    print(f"  final_diff ~ score_diff_mean: slope={beta_cal[1]:+.3f} (1.0=calibrated), "
          f"R2={r2_cal:.4f}, RMSE={rmse_s:.2f} pts")
    print(f"  (equity RMSE vs final_diff, off-scale, informational: {rmse_e:.2f})")
    print()

    move_h = r2_h - r2_base
    move_n = r2_n - r2_base
    if move_n > move_h:
        print(f"VERDICT: the model adds more move-quality signal than equity "
              f"(dR2 {move_n:+.4f} vs {move_h:+.4f}). The value estimate is sound; "
              f"the large-k loss is distribution/selection -> the PI loop is the fix.")
    else:
        print(f"VERDICT: equity adds at least as much as the model "
              f"(dR2 {move_h:+.4f} vs {move_n:+.4f}). The model is not yet a better "
              f"value estimate -> fix the model (more data / training) before the loop.")
    if abs(t_nh[2]) > 2.0:
        print("  Note: equity stays significant after the model -> the model is missing "
              "signal HastyBot has (most likely the leave term).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
