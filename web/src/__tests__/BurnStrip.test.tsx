import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import BurnStrip, { fmtUptime, type Fleet, type FleetInstance } from '../components/master/BurnStrip';

// The burn strip's three faces: loud while anything bills (with a way to the
// task that owns each instance), quiet when nothing does, amber when the
// listing it reads has stopped refreshing.

const getJSON = vi.fn();
vi.mock('../lib/api', () => ({
  getJSON: (...a: unknown[]) => getJSON(...a),
  postJSON: vi.fn(),
}));
vi.mock('../components/master/TaskView', () => ({ default: () => null }));

const now = () => Date.now() / 1000;

const inst = (over: Partial<FleetInstance>): FleetInstance => ({
  instance_id: 'i-1', type_id: 'g6.2xlarge', state: 'running', owner: 'position_eval/run7/m1',
  tracked: true, spot: false, cost_per_hr: 1.0, uptime_s: 5400, ...over,
});

const fleet = (over: Partial<Fleet>): Fleet => ({
  observed_at: now(), error: null, instances: [], burn_per_hr: 0, ...over,
});

const setup = async (f: Fleet, onOpen = vi.fn()) => {
  getJSON.mockResolvedValue(f);
  render(<BurnStrip onOpen={onOpen} />);
  await waitFor(() => expect(getJSON).toHaveBeenCalledWith('/api/cloud/fleet'));
  await waitFor(() => screen.getByRole('status'));
  return onOpen;
};

describe('the cloud burn strip', () => {
  beforeEach(() => getJSON.mockReset());

  it('adds up what bills and opens the task that owns an instance', async () => {
    const onOpen = await setup(fleet({
      burn_per_hr: 1.5,
      instances: [
        inst({}),
        inst({ instance_id: 'i-2', type_id: 'c7a.4xlarge', cost_per_hr: 0.5, owner: 'position_eval/run7/m2', uptime_s: 120 }),
      ],
    }));
    await waitFor(() => screen.getByText('burning $1.50/hr'));
    expect(screen.getByText('g6.2xlarge')).toBeInTheDocument();
    expect(screen.getByText('1h30')).toBeInTheDocument();
    fireEvent.click(screen.getAllByText('position_eval/run7')[0]);
    expect(onOpen).toHaveBeenCalledWith('position_eval', 'run7');
  });

  it('names an instance no task tracks as an orphan', async () => {
    await setup(fleet({ burn_per_hr: 0.5, instances: [inst({ tracked: false, owner: 'position_eval/gone/m1' })] }));
    await waitFor(() => screen.getByText('orphan · no task tracks it'));
    expect(screen.queryByText('position_eval/gone')).toBeNull();
  });

  it('is quiet when nothing bills, counting stopped disks', async () => {
    await setup(fleet({ instances: [inst({ state: 'stopped' })] }));
    await waitFor(() => screen.getByText('nothing billing'));
    expect(screen.getByText('· 1 stopped (disk only)')).toBeInTheDocument();
    expect(screen.queryByText('g6.2xlarge')).toBeNull();
  });

  it('says so when the listing has stopped refreshing', async () => {
    await setup(fleet({ observed_at: now() - 600 }));
    await waitFor(() => screen.getByText(/listing stale/));
  });

  it('shows why a listing failed instead of a zero it cannot vouch for', async () => {
    await setup(fleet({ observed_at: null, error: 'No credentials file at /x' }));
    await waitFor(() => screen.getByText('never listed'));
    expect(screen.getByText(/listing failed: No credentials/)).toBeInTheDocument();
  });

  it('formats uptime by the scale that matters', () => {
    expect(fmtUptime(30)).toBe('1m');
    expect(fmtUptime(2 * 3600 + 5 * 60)).toBe('2h05');
    expect(fmtUptime(3 * 86400)).toBe('3d');
  });
});
