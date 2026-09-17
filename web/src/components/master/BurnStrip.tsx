import { useEffect, useState } from 'react';
import { getJSON } from '../../lib/api';
import { relTime } from './MasterApp';

// The cloud burn strip: what the provider bills right now, pinned to the top
// of every dashboard page. Fed by GET /api/cloud/fleet -- the reconcile
// pass's listing of every instance tagged ours, whichever task (or none)
// names it -- so a machine left running under a tag you moved on from, or
// one whose task record was lost, shows here all the same. Loud when
// anything bills, quiet when nothing does, and amber when the listing itself
// has stopped refreshing: a dashboard that silently stopped observing is the
// one failure a quiet strip could otherwise hide.

export type FleetInstance = {
  instance_id: string; type_id: string; state: string;
  owner: string | null;  // "<workload>/<tag>/<machine>", the ownership tag; null if untagged
  tracked: boolean;  // some task's machines name it (else it is an orphan)
  spot: boolean; cost_per_hr: number | null; uptime_s: number | null;
};
export type Fleet = {
  observed_at: number | null;  // when the listing was taken; null before the first
  error: string | null;  // why the last listing failed, if it did
  instances: FleetInstance[];
  burn_per_hr: number;  // the rates of the pending/running instances, summed
};

const POLL_MS = 10000;
// A listing older than this means the reconcile pass has stopped observing;
// the strip says so rather than showing a zero it can no longer vouch for.
const STALE_S = 60;
const BILLING = new Set(['pending', 'running']);

export function fmtUptime(s: number | null): string {
  if (s == null) return '—';
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))}m`;
  const h = Math.floor(s / 3600);
  return h < 48 ? `${h}h${String(Math.round((s % 3600) / 60)).padStart(2, '0')}` : `${Math.round(h / 24)}d`;
}

const money = (x: number) => `$${x.toFixed(2)}`;

function InstanceChip({ inst, onOpen }: {
  inst: FleetInstance; onOpen: (workload: string, tag: string) => void;
}) {
  const billing = BILLING.has(inst.state);
  const [workload, tag, machine] = inst.owner?.split('/') ?? [];
  const canOpen = inst.tracked && workload && tag;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12.5,
      padding: '2px 9px', borderRadius: 999, background: 'white',
      border: `1px solid ${billing ? '#e0a0a0' : '#d3d8e0'}`, color: billing ? '#1a1f28' : '#7c8694',
    }}>
      <b>{inst.type_id}{inst.spot ? ' spot' : ''}</b>
      <span>{inst.state}{billing ? '' : ' (disk only)'}</span>
      {billing && <span>{fmtUptime(inst.uptime_s)}</span>}
      <span>{inst.cost_per_hr != null ? `${money(inst.cost_per_hr)}/hr` : 'rate unknown'}</span>
      {canOpen ? (
        <span
          onClick={() => onOpen(workload, tag)}
          title={`open ${workload}/${tag} (machine ${machine})`}
          style={{ color: '#1f77b4', cursor: 'pointer' }}
        >
          {workload}/{tag}
        </span>
      ) : (
        <span style={{ color: '#a05a00' }} title={inst.owner ?? 'no owner tag'}>orphan · no task tracks it</span>
      )}
    </span>
  );
}

export default function BurnStrip({ onOpen }: { onOpen: (workload: string, tag: string) => void }) {
  const [fleet, setFleet] = useState<Fleet | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  useEffect(() => {
    const poll = () => getJSON('/api/cloud/fleet')
      .then((f: Fleet) => { setFleet(f); setUnreachable(false); })
      .catch(() => setUnreachable(true));
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, []);

  const billing = fleet?.instances.filter((i) => BILLING.has(i.state)) ?? [];
  const idle = fleet?.instances.filter((i) => !BILLING.has(i.state)) ?? [];
  const burning = billing.length > 0;
  const stale = fleet != null
    && (fleet.observed_at == null || Date.now() / 1000 - fleet.observed_at > STALE_S);
  const warn = unreachable || stale || fleet?.error != null;
  const tone = burning ? '#b23b3b' : warn ? '#a05a00' : '#7c8694';
  return (
    <div style={{ position: 'sticky', top: 0, zIndex: 5, background: '#f4f6f8', padding: '4px 0 8px' }}>
      <div
        role="status"
        aria-label="cloud burn rate"
        style={{
          display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, fontSize: 13,
          padding: burning ? '7px 12px' : '3px 0', borderRadius: 8, color: tone,
          background: burning ? '#fdecec' : 'transparent',
          border: burning ? '1px solid #e0a0a0' : 'none',
        }}
      >
        <b style={{ fontSize: burning ? 15 : 13, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          Cloud
        </b>
        {fleet == null ? (
          <span>{unreachable ? 'dashboard API unreachable' : 'connecting…'}</span>
        ) : (
          <>
            <b style={{ fontSize: burning ? 15 : 13 }}>
              {burning ? `burning ${money(fleet.burn_per_hr)}/hr` : 'nothing billing'}
            </b>
            {fleet.instances.length === 0 && <span>· no instances</span>}
            {!burning && idle.length > 0 && (
              <span>· {idle.length} stopped (disk only)</span>
            )}
            {(burning ? fleet.instances : []).map((i) => (
              <InstanceChip key={i.instance_id} inst={i} onOpen={onOpen} />
            ))}
            {unreachable && <span className="pill-stale">dashboard API unreachable</span>}
            {fleet.error != null && <span className="pill-stale" title={fleet.error}>listing failed: {fleet.error}</span>}
            {stale && !unreachable && (
              <span className="pill-stale">
                {fleet.observed_at == null ? 'never listed' : `listing stale · ${relTime(fleet.observed_at)}`}
              </span>
            )}
            {!warn && <span>· listed {relTime(fleet.observed_at)}</span>}
          </>
        )}
      </div>
    </div>
  );
}
