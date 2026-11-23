import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import RunningStrip, { type RunningTag } from '../components/master/RunningStrip';

// The running strip: every tag with a running worker, whatever workload the
// page is on, each linking to its task view.

const getJSON = vi.fn();
vi.mock('../lib/api', () => ({ getJSON: (...a: unknown[]) => getJSON(...a) }));
vi.mock('../components/master/TaskView', () => ({ default: () => null }));

const setup = async (tags: RunningTag[], onOpen = vi.fn()) => {
  getJSON.mockResolvedValue({ tags });
  render(<RunningStrip onOpen={onOpen} />);
  await waitFor(() => expect(getJSON).toHaveBeenCalledWith('/api/running_tags'));
  return onOpen;
};

describe('the running strip', () => {
  beforeEach(() => {
    getJSON.mockReset();
  });

  it('lists each running tag with its running roles and opens it', async () => {
    const onOpen = await setup([
      {
        workload: 'position_eval', tag: 'run7',
        roles: [{ role: 'generate', title: 'Generator', running: 3 }, { role: 'train', title: 'Trainer (GPU)', running: 1 }],
      },
      { workload: 'blind_spots', tag: 'b2', roles: [{ role: 'generate', title: 'Surveyor', running: 2 }] },
    ]);
    await waitFor(() => screen.getByText('position_eval/run7'));
    expect(screen.getByText('3× Generator · 1× Trainer (GPU)')).toBeInTheDocument();
    expect(screen.getByText('2× Surveyor')).toBeInTheDocument();
    fireEvent.click(screen.getByText('blind_spots/b2'));
    expect(onOpen).toHaveBeenCalledWith('blind_spots', 'b2');
  });

  it('says so when nothing runs', async () => {
    await setup([]);
    await waitFor(() => screen.getByText('nothing running'));
  });

  it('says so when the listing cannot be fetched', async () => {
    getJSON.mockRejectedValue(new Error('502'));
    render(<RunningStrip onOpen={vi.fn()} />);
    await waitFor(() => screen.getByText('could not list running tags'));
  });
});
