import { useEffect, useState } from 'react';
import BokehFigure from './BokehFigure';
import GenerationSlider from './GenerationSlider';
import { getJSON } from '../lib/api';

// A dashboard tab that embeds a per-generation Bokeh figure (Positions, Calibration)
// whose generation is chosen by the shared React GenerationSlider -- re-fetching the
// figure per generation -- instead of an in-figure Bokeh slider. (The Positions
// figure keeps its own position scrubber; it resets when the generation changes.)
export default function GenerationFigureTab({
  task,
  tag,
  figure,
  genTable,
  emptyText,
}: {
  task: string;
  tag: string | null;
  figure: string;
  genTable: string;
  emptyText: string;
}) {
  const [generations, setGenerations] = useState<number[]>([]);
  const [genIdx, setGenIdx] = useState(0);
  const [latest, setLatest] = useState(true);
  const [item, setItem] = useState<unknown | null>(null);

  // Poll the generation (epoch) list; keep the same array when unchanged so an
  // idle poll doesn't re-render or refetch.
  useEffect(() => {
    if (!tag) {
      setGenerations([]);
      return;
    }
    const refresh = () =>
      getJSON(`/api/generations?task=${task}&tag=${tag}&table=${genTable}`)
        .then((d: { generations: number[] }) =>
          setGenerations((prev) => {
            const next = d.generations;
            const same =
              prev.length === next.length && prev[prev.length - 1] === next[next.length - 1];
            return same ? prev : next;
          }),
        )
        .catch(() => {});
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [task, tag, genTable]);

  const genCount = generations.length;
  // Follow the newest generation while "Latest" is checked.
  useEffect(() => {
    if (latest) setGenIdx(Math.max(0, genCount - 1));
  }, [latest, genCount]);
  const effIdx = Math.min(genIdx, Math.max(0, genCount - 1));

  // Fetch the figure for the selected generation.
  useEffect(() => {
    if (!tag || genCount === 0) {
      setItem(null);
      return;
    }
    getJSON(`/api/figure/${figure}?task=${task}&tag=${tag}&gen_idx=${effIdx}`)
      .then((d) => setItem(d.item ?? null))
      .catch(() => setItem(null));
  }, [task, tag, figure, genCount, effIdx]);

  if (genCount === 0) {
    return (
      <div className="card">
        <div style={{ color: '#556070', fontStyle: 'italic', padding: 20 }}>{emptyText}</div>
      </div>
    );
  }

  return (
    <div className="card">
      <div style={{ marginBottom: 10, fontSize: 15, display: 'flex', alignItems: 'center', gap: 12 }}>
        <span>Generation</span>
        <GenerationSlider
          count={genCount}
          valueIdx={effIdx}
          follow={latest}
          onChange={(idx, f) => {
            setGenIdx(idx);
            setLatest(f);
          }}
          label={`epoch ${generations[effIdx]}`}
        />
      </div>
      {item ? (
        <BokehFigure item={item} />
      ) : (
        <div style={{ color: '#556070', padding: 20 }}>Loading…</div>
      )}
    </div>
  );
}
