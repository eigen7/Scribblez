"""FP16-safety export gate (docs/fp16_safe_serving.md).

Activation magnitudes grow monotonically under an unpenalized recipe, and a
checkpoint whose intermediates cross FP16 range NaNs deterministically once
the engine serves it in FP16. The gate makes FP16-safety a property of the
exported model rather than the runtime: an export runs a probe batch through
the just-written ONNX graph in FP32 with every intermediate tensor exposed,
and fails if any intermediate comes within 4x of FP16's max normal. The
measured peaks are also the observability: a trainer records the per-export
peak, giving the growth curve of each run for free.

This module is family-agnostic: it probes any ONNX file given ready feeds.
Each family builds its own probe feeds next to its exporter (extreme
current-score leads are where magnitudes peak, so the builders sweep real
positions across PROBE_LEADS on top of the rows' own differentials).
"""

import operator
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

FP16_MAX = 65504.0

# The gate threshold: FP16 range with 4x headroom. A served checkpoint should
# not merely fit under 65504, it should sit far from the cliff -- the incident
# checkpoint crossed the range mid-run, so "fits today" is not a property worth
# certifying (docs/fp16_safe_serving.md).
GATE_THRESHOLD = 16384.0

# Current-score leads (points) the probe-feed builders stamp onto real rows,
# on top of the rows' own leads. The incident NaNs appeared at +150..+190;
# sweeping past +-300 probes beyond anything rollouts reach.
PROBE_LEADS = (-300, -200, -100, 100, 200, 300)


class Fp16HeadroomError(RuntimeError):
    """An exported graph's intermediates exceed the FP16 headroom threshold.

    `peaks` holds every offending (tensor_name, peak_abs) pair, worst first.
    """

    def __init__(self, path, peaks: list[tuple[str, float]], threshold: float):
        self.peaks = peaks
        listing = ", ".join(f"{name}={peak:.0f}" for name, peak in peaks[:8])
        more = f" (+{len(peaks) - 8} more)" if len(peaks) > 8 else ""
        super().__init__(
            f"{path}: {len(peaks)} intermediate tensor(s) exceed the FP16 headroom "
            f"threshold {threshold:.0f} (FP16 max {FP16_MAX:.0f}): {listing}{more}. "
            f"This checkpoint would overflow when served in FP16; see "
            f"docs/fp16_safe_serving.md."
        )


def _probe_session(path: Path) -> ort.InferenceSession:
    """A CPU session over the exported graph with every intermediate tensor
    promoted to a graph output. Graph optimizations are disabled so the
    measured tensors are the graph as written -- an optimizer could fold away
    exactly the node being measured. onnx.load resolves external data (the
    shared lexicon blob) against the file's own directory, so serializing the
    modified model to bytes is self-contained."""
    model = onnx.load(str(path))
    inferred = onnx.shape_inference.infer_shapes(model)
    existing = {o.name for o in inferred.graph.output}
    initializers = {i.name for i in inferred.graph.initializer}
    for vi in inferred.graph.value_info:
        if vi.name not in existing and vi.name not in initializers:
            inferred.graph.output.append(vi)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        inferred.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def intermediate_peaks(path: str | Path, feeds: list[dict]) -> list[tuple[str, float]]:
    """Peak |value| per graph tensor over `feeds`, worst first.

    Each feed is one {input_name: ndarray} dict run separately, so callers
    chunk large probes to bound memory (every intermediate of every row is
    materialized as an output). Non-float tensors (indices, masks) are skipped.
    """
    sess = _probe_session(Path(path))
    names = [o.name for o in sess.get_outputs()]
    peaks: dict[str, float] = {}
    for feed in feeds:
        for name, value in zip(names, sess.run(None, feed), strict=True):
            if not np.issubdtype(value.dtype, np.floating) or value.size == 0:
                continue
            peak = float(np.abs(value).max())
            if peak > peaks.get(name, 0.0):
                peaks[name] = peak
    return sorted(peaks.items(), key=operator.itemgetter(1), reverse=True)


def check_fp16_headroom(
    path: str | Path, feeds: list[dict], *, threshold: float = GATE_THRESHOLD
) -> float:
    """The gate: probe `path` over `feeds` and raise Fp16HeadroomError if any
    intermediate exceeds `threshold`. Returns the overall peak |activation|
    (the scalar a trainer records per export)."""
    peaks = intermediate_peaks(path, feeds)
    offenders = [(name, peak) for name, peak in peaks if peak > threshold]
    if offenders:
        raise Fp16HeadroomError(path, offenders, threshold)
    return peaks[0][1] if peaks else 0.0
