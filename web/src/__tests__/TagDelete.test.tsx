import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { HomePage, type Workload } from '../components/master/MasterApp';

// What the tag listing's Delete control is gated on: worker slots the operator
// has running, not slots outright -- deleting a tag releases its idle ones.

const getJSON = vi.fn();
const postJSON = vi.fn();
vi.mock('../lib/api', () => ({
  getJSON: (...a: unknown[]) => getJSON(...a),
  postJSON: (...a: unknown[]) => postJSON(...a),
}));
vi.mock('../components/master/TaskView', () => ({ default: () => null }));

const workload: Workload = {
  name: 'position_eval', title: 'Train position evaluation',
  roles: [], primary_params: [], params: [],
};

const row = (tag: string, workers: number, active_workers: number) => ({
  tag, has_task: true, created_at: 0, workers, active_workers,
  progress: [] as [string, string | number][], last_active: 0,
});

const deleteButton = (tag: string) =>
  screen.getByText(tag).closest('tr')!.querySelector('button') as HTMLButtonElement;

const setup = async (...rows: ReturnType<typeof row>[]) => {
  getJSON.mockResolvedValue({ tags: rows });
  render(<HomePage workload={workload} onOpen={() => {}} />);
  await waitFor(() => screen.getByText(rows[0].tag));
};

describe('the tag listing’s Delete control', () => {
  beforeEach(() => {
    getJSON.mockReset();
    postJSON.mockReset();
    vi.spyOn(window, 'confirm').mockReturnValue(true);
  });

  it('deletes a tag whose worker slots are all paused', async () => {
    postJSON.mockResolvedValue({ ok: true });
    await setup(row('idle', 2, 0));

    expect(deleteButton('idle').disabled).toBe(false);
    fireEvent.click(deleteButton('idle'));
    await waitFor(() =>
      expect(postJSON).toHaveBeenCalledWith('/api/task/delete', {
        workload: 'position_eval', tag: 'idle',
      }),
    );
  });

  it('says the slots are going too, so the confirmation is not a surprise', async () => {
    await setup(row('idle', 2, 0));
    fireEvent.click(deleteButton('idle'));
    expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('2 idle worker slots'));
  });

  it('refuses a tag with a running worker', async () => {
    await setup(row('busy', 3, 1));
    expect(deleteButton('busy').disabled).toBe(true);
  });
});
