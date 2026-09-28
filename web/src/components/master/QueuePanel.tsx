import { useEffect, useState } from 'react';
import { getJSON, postJSON } from '../../lib/api';
import { Button, Workload } from './MasterApp';
import { enqueueTag } from './PoolView';

// What the tag queue would start for this tag, shown before and after it is
// enqueued, with Enqueue or Dequeue beside it. The roles follow from the tag's
// params through the workload's layout; per machine, the slots it would get
// there and why it cannot take the tag, if so.
type QueuePlan = {
  roles: string[];
  machines: {
    machine: string; slots: { role: string; threads: number | null }[];
    gpu_gb: number | null; refusal: string | null;
  }[];
};

export function QueuePanel({ workload, tag, queued, onChanged }: {
  workload: Workload; tag: string; queued: number | null; onChanged: () => void;
}) {
  const [plan, setPlan] = useState<QueuePlan | null>(null);
  useEffect(() => {
    // A workload with no layout is not queueable: no panel.
    getJSON(`/api/queue/plan?workload=${encodeURIComponent(workload.name)}&tag=${encodeURIComponent(tag)}`)
      .then(setPlan)
      .catch(() => setPlan(null));
  }, [workload.name, tag, queued]);
  if (!plan) return null;
  const title = (role: string) => workload.roles.find((r) => r.name === role)?.title ?? role;
  const onError = (e: Error) => window.alert(String(e.message ?? e));
  return (
    <div style={{ fontSize: 13, color: '#556070', marginBottom: 10 }} data-testid="queue-panel">
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        {queued != null ? (
          <>
            <span><b>queued #{queued}</b> (Machine pool page)</span>
            <Button label="Dequeue" tone="danger" onClick={() => {
              postJSON('/api/queue/action', { workload: workload.name, tag, action: 'dequeue' })
                .then(onChanged).catch(onError);
            }} />
          </>
        ) : (
          <span title="place this tag on the first free pool machine that fits it (Machine pool page)">
            <Button label="Enqueue" onClick={() => {
              enqueueTag(workload.name, tag).then(onChanged).catch(onError);
            }} />
          </span>
        )}
        <span>
          {queued != null ? 'will run' : 'would run'} <b>{plan.roles.map(title).join(' + ')}</b>, as
          this tag's params ask
        </span>
      </div>
      {plan.machines.length === 0 ? (
        <div style={{ marginTop: 4 }}>The pool has no machines or rental capacity yet.</div>
      ) : plan.machines.map((m) => (
        <div key={m.machine} style={{ marginTop: 4 }}>
          {m.machine}: {m.slots.map((sl) => title(sl.role) + (sl.threads ? ` (${sl.threads} threads)` : '')).join(', ')}
          {m.gpu_gb != null && `; ${m.gpu_gb.toFixed(1)} GiB GPU`}
          {m.refusal && <span style={{ color: '#a05a00' }}> — cannot take it: {m.refusal}</span>}
        </div>
      ))}
    </div>
  );
}
