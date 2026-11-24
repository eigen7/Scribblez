import { useEffect, useState } from 'react';
import OppRackBelief from './OppRackBelief';
import { getJSON } from '../lib/api';
import { RackBeliefChart } from '../lib/surveyTypes';

// The leaf Positions tab's opponent panel: the mover's belief about the
// opponent's rack at one known position under the tag's fixed leave inference
// model, the one its games draw the opponent's leaves from
// (/api/deep_bot_leaf_eval/belief), so the same whatever generation the tab
// shows. The tag's first request charts every known position on the
// controller, about a second.

export default function LeafOppBelief({
  task, tag, position,
}: {
  task: string;
  tag: string;
  position: number;
}) {
  const [chart, setChart] = useState<RackBeliefChart | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setChart(null);
    setError(null);
    getJSON(`/api/deep_bot_leaf_eval/belief?task=${task}&tag=${tag}&position=${position}`)
      .then((d: { belief: RackBeliefChart }) => { if (!cancelled) setChart(d.belief); })
      .catch((e: Error) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [task, tag, position]);
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
