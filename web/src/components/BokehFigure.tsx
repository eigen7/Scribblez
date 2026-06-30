import { useEffect, useRef } from 'react';
import * as Bokeh from '@bokeh/bokehjs';

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
export default function BokehFigure({ item }: { item: unknown | null }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.replaceChildren();
    if (!item) return;

    let cancelled = false;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let views: any = null;
    const target = document.createElement('div');
    el.appendChild(target);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Bokeh.embed
      .embed_item(item as any, target)
      .then((v: unknown) => {
        // If we've already unmounted by the time it resolves, tear it down now.
        if (cancelled) (v as { clear?: () => void })?.clear?.();
        else views = v;
      })
      .catch((e: unknown) => {
        if (!cancelled) console.error('Bokeh embed failed', e);
      });

    return () => {
      cancelled = true;
      try {
        views?.clear?.();
      } catch {
        /* already torn down */
      }
      el.replaceChildren();
    };
  }, [item]);

  return <div ref={ref} />;
}
