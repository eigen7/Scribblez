import { describe, expect, it } from 'vitest';
import {
  AnyModel, applyRanges, applySourceTails, plotsOf, rowsOf, sourceStates,
} from '../lib/bokehDoc';

// A stub range/CDS/plot/document shaped like the walked slice of BokehJS.
function range() {
  return {
    start: 0, end: 1,
    setv(attrs: { start: number; end: number }) { this.start = attrs.start; this.end = attrs.end; },
  };
}

function source(name: string | null, x: number[], y: number[]) {
  return {
    name, data: { x, y },
    streamed: [] as unknown[],
    stream(cols: Record<string, number[]>) {
      this.streamed.push(cols);
      this.data.x.push(...cols.x); this.data.y.push(...cols.y);
    },
  };
}

function plot(sources: AnyModel[]) {
  return { x_range: range(), y_range: range(), renderers: sources.map((s) => ({ data_source: s })) };
}

function stubViews() {
  // Two named rows sharing source names, as the Loss tab's figures are built.
  const linA = source('Loss|a', [1, 2], [10, 20]);
  const logA = source('Loss|a', [1, 2], [10, 20]);
  const rows = [
    { name: 'x_linear', children: [plot([linA])] },
    { name: 'x_log', children: [plot([logA])] },
  ];
  return { views: { roots: [{ model: { children: rows } }] }, linA, logA };
}

describe('bokehDoc', () => {
  it('collects plots and named rows', () => {
    const { views } = stubViews();
    expect(plotsOf(views)).toHaveLength(2);
    expect(Object.keys(rowsOf(views))).toEqual(['x_linear', 'x_log']);
  });

  it('reads one cursor per name and null when nothing is named', () => {
    const { views } = stubViews();
    expect(sourceStates(plotsOf(views))).toEqual({ 'Loss|a': { n: 2, last_x: 2 } });
    expect(sourceStates([plot([source(null, [1], [1])])])).toBeNull();
  });

  it('streams a tail into every copy of a named source, skipping empty tails', () => {
    const { views, linA, logA } = stubViews();
    applySourceTails(plotsOf(views), { 'Loss|a': { x: [3], y: [30] } });
    expect(linA.data.x).toEqual([1, 2, 3]);
    expect(logA.data.x).toEqual([1, 2, 3]);
    applySourceTails(plotsOf(views), { 'Loss|a': { x: [], y: [] } });
    expect(linA.streamed).toHaveLength(1); // empty tail -> no stream call
  });

  it('applies explicit ranges per row and honors the zoom guard', () => {
    const { views } = stubViews();
    const rows = rowsOf(views);
    const logPlot = rows.x_log.children[0];
    applyRanges(rows, {
      x_linear: [{ x: null, y: null }],
      x_log: [{ x: [5, 50], y: null }],
    }, () => true);
    expect([logPlot.x_range.start, logPlot.x_range.end]).toEqual([5, 50]);
    expect(rows.x_linear.children[0].x_range.start).toBe(0); // null -> untouched
    applyRanges(rows, { x_log: [{ x: [7, 70], y: null }] }, () => false);
    expect(logPlot.x_range.start).toBe(5); // guard blocked the update
  });
});
