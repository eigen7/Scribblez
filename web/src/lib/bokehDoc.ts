// Walking and incrementally updating an embedded Bokeh document.
//
// The dashboard's figures are built server-side and embedded whole; when the run
// advances, the client streams just the appended rows into the live document
// instead of refetching and re-embedding it (the server counterpart, protocol
// included, is py/scribblez/dashboard/figure_delta.py). The helpers here walk
// plain model objects and are BokehJS-import-free, so they are unit-testable
// against stub documents.

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyModel = any;

// One data source's cursor as the server expects it: how many points the client
// holds and the last x among them (a mismatch there means history was rewritten).
export interface SourceState { n: number; last_x: number | null }

// A panel's explicit axis ranges from the server; null means the axis is
// auto-ranged and BokehJS follows the streamed data itself.
export interface PanelRanges { x: [number, number] | null; y: [number, number] | null }

// The /api/figure_delta response.
export interface FigureDelta {
  refetch?: boolean;
  sources?: Record<string, Record<string, number[]>>;
  ranges?: Record<string, PanelRanges[]>;
}

// What BokehFigure exposes for the incremental path: read the embedded
// document's cursors, and apply a delta to it in place.
export interface FigureHandle {
  sourceStates(): Record<string, SourceState> | null;
  applyDelta(delta: FigureDelta): void;
}

// Collect Plot models (those carrying x/y ranges) from a model tree in layout
// order. The figure is rebuilt identically each refresh, so plot i before a
// refresh corresponds to plot i after it.
export function collectPlots(model: AnyModel, out: AnyModel[]) {
  if (!model || typeof model !== 'object') return;
  if (model.x_range && model.y_range) {
    out.push(model);
    return;
  }
  if (Array.isArray(model.children)) {
    for (const k of model.children) collectPlots(Array.isArray(k) ? k[0] : k, out);
  } else if (Array.isArray(model.tabs)) {
    for (const t of model.tabs) collectPlots(t?.child, out);
  } else if (model.child) {
    collectPlots(model.child, out);
  }
}

export function plotsOf(views: AnyModel): AnyModel[] {
  const out: AnyModel[] = [];
  try {
    for (const rootView of views?.roots ?? []) collectPlots(rootView.model, out);
  } catch {
    /* unexpected shape -> just skip */
  }
  return out;
}

// The named layout rows directly under the embed's roots (the x-axis variant
// rows the server names via plots.py's X_AXIS_LINEAR / X_AXIS_LOG).
export function rowsOf(views: AnyModel): Record<string, AnyModel> {
  const out: Record<string, AnyModel> = {};
  try {
    for (const rootView of views?.roots ?? []) {
      for (const child of rootView.model?.children ?? []) {
        if (child?.name) out[child.name] = child;
      }
    }
  } catch {
    /* unexpected shape -> just skip */
  }
  return out;
}

// name -> every data source carrying that name across `plots` (the linear and
// log rows share names by design; each copy holds identical data).
export function namedSources(plots: AnyModel[]): Record<string, AnyModel[]> {
  const out: Record<string, AnyModel[]> = {};
  for (const p of plots) {
    for (const r of p.renderers ?? []) {
      const src = r?.data_source;
      if (src?.name) (out[src.name] ??= []).push(src);
    }
  }
  return out;
}

// The cursors to post to the delta endpoint, or null when the document carries
// no named sources (not built for streaming) -- the caller then full-refetches.
export function sourceStates(plots: AnyModel[]): Record<string, SourceState> | null {
  const sources = namedSources(plots);
  const names = Object.keys(sources);
  if (names.length === 0) return null;
  const out: Record<string, SourceState> = {};
  for (const name of names) {
    const x = sources[name][0].data?.x ?? [];
    out[name] = { n: x.length, last_x: x.length ? Number(x[x.length - 1]) : null };
  }
  return out;
}

// Stream each source's appended rows into every copy of it.
export function applySourceTails(plots: AnyModel[], tails: Record<string, Record<string, number[]>>) {
  const sources = namedSources(plots);
  for (const [name, cols] of Object.entries(tails)) {
    if (!Object.values(cols).some((v) => v.length)) continue;
    for (const src of sources[name] ?? []) src.stream(cols);
  }
}

// Move each panel's explicit ranges to the server's current extents, per named
// row in panel layout order. `mayAdjust` lets the caller protect panels the
// user has zoomed.
export function applyRanges(
  rows: Record<string, AnyModel>,
  ranges: Record<string, PanelRanges[]>,
  mayAdjust: (plot: AnyModel) => boolean,
) {
  for (const [name, panels] of Object.entries(ranges)) {
    const plots: AnyModel[] = [];
    collectPlots(rows[name], plots);
    panels.forEach((r, i) => {
      const p = plots[i];
      if (!p || !mayAdjust(p)) return;
      if (r.x) p.x_range.setv({ start: r.x[0], end: r.x[1] });
      if (r.y) p.y_range.setv({ start: r.y[0], end: r.y[1] });
    });
  }
}
