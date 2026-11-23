import { useEffect, useState } from 'react';
import { getJSON } from '../../lib/api';
import { TagLink } from './ui';

// The running strip: every tag, of any workload, with a worker running now,
// beneath the burn strip atop every dashboard page, so what is running is in
// view whichever workload the selector is on. Fed by GET /api/running_tags,
// read from the reconcile pass's last observations: "running" here is the
// state the task view shows a slot in, not merely a slot meant to run.

export type RunningTag = {
  workload: string; tag: string;
  roles: { role: string; title: string; running: number }[];  // roles with none are left out
};

const POLL_MS = 5000;

function TagChip({ t, onOpen }: { t: RunningTag; onOpen: (workload: string, tag: string) => void }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12.5,
      padding: '2px 9px', borderRadius: 999, background: 'white', border: '1px solid #b5d6b5',
    }}>
      <TagLink workload={t.workload} tag={t.tag} onOpen={onOpen} />
      <span style={{ color: '#556070' }}>
        {t.roles.map((r) => `${r.running}× ${r.title}`).join(' · ')}
      </span>
    </span>
  );
}

export default function RunningStrip({ onOpen }: { onOpen: (workload: string, tag: string) => void }) {
  const [tags, setTags] = useState<RunningTag[] | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const poll = () => getJSON('/api/running_tags')
      .then((d: { tags: RunningTag[] }) => { setTags(d.tags); setError(''); })
      .catch((e) => setError(String(e)));
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, []);

  const running = tags != null && tags.length > 0;
  return (
    <div
      role="status"
      aria-label="running tags"
      style={{
        display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, fontSize: 13,
        marginBottom: 10, color: running ? '#2a7a2a' : '#7c8694',
      }}
    >
      <b style={{ textTransform: 'uppercase', letterSpacing: '0.04em' }}>Running</b>
      {tags == null ? (
        <span>{error ? 'could not list running tags' : 'connecting…'}</span>
      ) : !running ? (
        <span>nothing running</span>
      ) : (
        tags.map((t) => <TagChip key={`${t.workload}/${t.tag}`} t={t} onOpen={onOpen} />)
      )}
      {error && tags != null && <span className="pill-stale" title={error}>listing failed</span>}
    </div>
  );
}
