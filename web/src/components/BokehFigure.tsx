import { MutableRefObject, useEffect, useRef } from 'react';
import * as Bokeh from '@bokeh/bokehjs';
import {
  AnyModel, applyRanges, applySourceTails, FigureDelta, FigureHandle, plotsOf, rowsOf,
  sourceStates,
} from '../lib/bokehDoc';
import { applyVisibility, Visibility } from '../lib/bokehVisibility';

// A Bokeh figure built server-side (bokeh.embed.json_item) and rendered here with
// BokehJS. This is how the React dashboard hosts the existing Bokeh plots without
// rewriting them: the Python data API returns the json_item, and embed_item mounts
// it into our div. CustomJS-driven interactivity (sliders, scrubbers) is carried
// inside the item and keeps working. The BokehJS version is pinned to match the
// Python bokeh that produced the item (see web/package.json).
//
// On unmount/replace the Bokeh views are destroyed via the ViewManager's clear()
// -- not just by emptying the DOM -- so floating overlays like a HoverTool tooltip
// don't linger after a tab switch.
//
// Zoom preservation: each data refresh re-embeds a fresh item, which would reset any
// pan/zoom. Before tearing the old figure down we snapshot each plot's current axis
// ranges and, if the user had zoomed away from the auto extent, re-apply them to the
// matching plot in the new figure. A range still at the auto extent (never touched,
// or reset via the toolbar) is left to keep following the incoming data.
//
// Incremental refresh: `handleRef`, when given, is filled with a FigureHandle
// exposing the embedded document to the streaming path (lib/bokehDoc.ts): read
// the per-source cursors, and apply a figure delta in place -- appended rows are
// streamed into every copy of each named source, and the explicit axis ranges
// are moved to the server's extents on panels the user has not zoomed. A range
// snapshot is refreshed after each apply, so the zoom preservation below keeps
// telling data-driven range growth apart from the user's own zooming.
//
// Named-model visibility: `visibility` maps model names inside the item (set
// server-side via Bokeh's `name`) to whether they are displayed. It is applied
// right after the embed resolves -- before the browser paints, so no flash -- and
// re-applied in place when it changes, so a control can flip between pre-built
// variants of a figure without a refetch or re-embed.
//
// Scroll preservation: the same teardown/re-embed would also momentarily collapse
// the page (the embed is async), making the browser clamp the scroll position to
// the shorter document. The container is pinned at its old height across the gap
// via min-height, released once the new figure has laid out.

type AxisRange = [number | null, number | null];
interface PlotRange {
  x: AxisRange;
  y: AxisRange;
}

function readRange(p: AnyModel): PlotRange | null {
  try {
    return { x: [p.x_range.start, p.x_range.end], y: [p.y_range.start, p.y_range.end] };
  } catch {
    return null;
  }
}

function applyRange(p: AnyModel, r: PlotRange) {
  try {
    if (r.x[0] != null && r.x[1] != null) p.x_range.setv({ start: r.x[0], end: r.x[1] });
    if (r.y[0] != null && r.y[1] != null) p.y_range.setv({ start: r.y[0], end: r.y[1] });
  } catch {
    /* ignore */
  }
}

function axisZoomed(cur: AxisRange, auto: AxisRange): boolean {
  const [cs, ce] = cur;
  const [as_, ae] = auto;
  if (cs == null || ce == null || as_ == null || ae == null) return false;
  const tol = (Math.abs(ae - as_) || 1) * 0.005;
  return Math.abs(cs - as_) > tol || Math.abs(ce - ae) > tol;
}

function isZoomed(cur: PlotRange, auto: PlotRange): boolean {
  return axisZoomed(cur.x, auto.x) || axisZoomed(cur.y, auto.y);
}

// The document an embed's views render, for named-model lookups.
function documentOf(views: AnyModel): AnyModel {
  return views?.roots?.[0]?.model?.document ?? null;
}

export default function BokehFigure({ item, visibility, handleRef }: {
  item: unknown | null; visibility?: Visibility; handleRef?: MutableRefObject<FigureHandle | null>;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const viewsRef = useRef<AnyModel>(null); // the live embed, for in-place updates
  const visibilityRef = useRef<Visibility | undefined>(visibility); // latest, for the embed
  const plotsRef = useRef<AnyModel[]>([]); // the current figure's plots, in order
  const autoRef = useRef<(PlotRange | null)[]>([]); // their auto extents at embed time
  const pinnedRef = useRef<(PlotRange | null)[]>([]); // user-zoomed ranges to restore

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.replaceChildren();
    if (!item) {
      el.style.minHeight = '';
      return;
    }

    let cancelled = false;
    let views: AnyModel = null;
    const target = document.createElement('div');
    el.appendChild(target);
    Bokeh.embed
      .embed_item(item as AnyModel, target)
      .then((v: AnyModel) => {
        if (cancelled) {
          v?.clear?.();
          return;
        }
        views = v;
        viewsRef.current = v;
        if (visibilityRef.current) applyVisibility(documentOf(v), visibilityRef.current);
        const plots = plotsOf(v);
        plotsRef.current = plots;
        if (handleRef) handleRef.current = makeHandle();
        // After the layout has computed its auto ranges, snapshot them and re-apply
        // whatever the user had zoomed to before this refresh.
        requestAnimationFrame(() => {
          if (cancelled) return;
          autoRef.current = plots.map(readRange);
          plots.forEach((p, i) => {
            const r = pinnedRef.current[i];
            if (r) applyRange(p, r);
          });
          el.style.minHeight = '';
        });
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        console.error('Bokeh embed failed', e);
        el.style.minHeight = '';
      });

    return () => {
      cancelled = true;
      // Hold the container at its current height across the teardown/re-embed gap
      // (the next effect's embed is async): letting the page collapse in between
      // makes the browser clamp the scroll position, yanking the user back to the
      // top on every data refresh. The next effect releases the hold once its
      // figure has laid out (or it has nothing to embed).
      el.style.minHeight = `${el.offsetHeight}px`;
      // Capture each plot's current view vs. its snapshotted auto extent; a range
      // still matching the auto extent (untouched or reset) is left to follow the
      // data, otherwise it is pinned for the next embed to restore.
      const plots = plotsRef.current;
      const autos = autoRef.current;
      pinnedRef.current = plots.map((p, i) => {
        const cur = readRange(p);
        const auto = autos[i];
        return cur && auto && isZoomed(cur, auto) ? cur : null;
      });
      try {
        views?.clear?.();
      } catch {
        /* already torn down */
      }
      viewsRef.current = null;
      if (handleRef) handleRef.current = null;
      el.replaceChildren();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [item]);

  // Keep the latest map for an embed still in flight, and apply it in place to
  // the live one.
  useEffect(() => {
    visibilityRef.current = visibility;
    if (visibility) applyVisibility(documentOf(viewsRef.current), visibility);
  }, [visibility]);

  // The streaming handle over the current embed (see the header comment).
  function makeHandle(): FigureHandle {
    return {
      sourceStates: () => (viewsRef.current ? sourceStates(plotsRef.current) : null),
      applyDelta: (delta: FigureDelta) => {
        const views = viewsRef.current;
        if (!views) return;
        const plots = plotsRef.current;
        // Panels the user has zoomed (view differs from the auto snapshot) keep
        // their view; everything else follows the server's ranges.
        const zoomedNow = plots.map((p, i) => {
          const cur = readRange(p);
          const auto = autoRef.current[i];
          return !!(cur && auto && isZoomed(cur, auto));
        });
        if (delta.sources) applySourceTails(plots, delta.sources);
        if (delta.ranges) {
          applyRanges(rowsOf(views), delta.ranges, (p) => {
            const i = plots.indexOf(p);
            return i < 0 || !zoomedNow[i];
          });
        }
        // Auto ranges recompute from the streamed data; re-snapshot them (for
        // panels the user left alone) once that has happened.
        requestAnimationFrame(() => {
          plots.forEach((p, i) => {
            if (!zoomedNow[i]) autoRef.current[i] = readRange(p);
          });
        });
      },
    };
  }

  return <div ref={ref} />;
}
