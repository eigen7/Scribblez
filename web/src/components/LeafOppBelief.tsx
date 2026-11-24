import { useEffect, useState } from 'react';
import OppRackBelief from './OppRackBelief';
import { getJSON } from '../lib/api';
import { RackBeliefChart } from '../lib/surveyTypes';

// The leaf Positions tab's opponent panel: the mover's belief about the
// opponent's rack at one known position under the leave inference model one
// generation's analysis drew the opponent's leaves from
// (/api/deep_bot_leaf_eval/belief). The first request for a generation
// charts every known position on the controller, about a second.

export default function LeafOppBelief({
  task, tag, position, generation,
}: {
  task: string;
  tag: string;
  position: number;
  generation: number;
}) {
  const [chart, setChart] = useState<RackBeliefChart | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setChart(null);
    setError(null);
    getJSON(`/api/deep_bot_leaf_eval/belief?task=${task}&tag=${tag}&position=${position}&generation=${generation}`)
      .then((d: { belief: RackBeliefChart }) => { if (!cancelled) setChart(d.belief); })
      .catch((e: Error) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [task, tag, position, generation]);
  if (error) return <div className="dbp-error">Opponent&apos;s rack: {error}</div>;
  if (!chart) {
    return (
      <div className="muted">
        <span className="scz-spinner" style={{ verticalAlign: 'middle' }} /> charting the opponent&apos;s rack…
      </div>
    );
  }
  return <OppRackBelief chart={chart} />;
}
