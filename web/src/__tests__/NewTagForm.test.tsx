import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { NewTagForm, type Workload } from '../components/master/MasterApp';

// The new-tag form's handling of a closed parameter: it starts unchosen so a
// run cannot inherit an arm nobody picked, blocks creation until it is picked,
// and re-seeds the defaults that choice implies for the other fields.

const postJSON = vi.fn();
vi.mock('../lib/api', () => ({ getJSON: vi.fn(), postJSON: (...a: unknown[]) => postJSON(...a) }));
vi.mock('../components/master/TaskView', () => ({ default: () => null }));

const workload: Workload = {
  name: 'position_eval',
  title: 'Train position evaluation',
  roles: [],
  params: [
    {
      name: 'optimizer', kind: 'str', default: 'wsd', help: 'the arm',
      choices: { wsd: { lr: 0.001 }, schedule_free: { lr: 0.0025 } },
    },
    { name: 'lr', kind: 'float', default: 0.001, help: 'the rate', choices: null },
    { name: 'batch_size', kind: 'int', default: 256, help: 'batch', choices: null },
  ],
};

const setup = () => {
  render(<NewTagForm workload={workload} onCreated={() => {}} />);
  fireEvent.change(screen.getByPlaceholderText('e.g. exp42'), { target: { value: 'tryit' } });
};

const optimizer = () => screen.getByLabelText('optimizer') as HTMLSelectElement;
const lr = () => screen.getByLabelText('lr') as HTMLInputElement;
const createButton = () => screen.getByText('Create').closest('button') as HTMLButtonElement;

describe('NewTagForm with a closed parameter', () => {
  beforeEach(() => postJSON.mockReset());

  it('renders it as a selector that starts unchosen', () => {
    setup();
    expect(optimizer().value).toBe('');
    expect([...optimizer().options].map((o) => o.value)).toEqual(['', 'wsd', 'schedule_free']);
  });

  it('blocks creation until it is chosen', () => {
    setup();
    expect(createButton().disabled).toBe(true);
    fireEvent.change(optimizer(), { target: { value: 'wsd' } });
    expect(createButton().disabled).toBe(false);
  });

  it('re-seeds the defaults the choice implies', () => {
    setup();
    fireEvent.change(optimizer(), { target: { value: 'schedule_free' } });
    expect(lr().value).toBe('0.0025');
    // ... and switching back re-seeds the other arm's rate.
    fireEvent.change(optimizer(), { target: { value: 'wsd' } });
    expect(lr().value).toBe('0.001');
  });

  it('submits the seeded values, coerced to their kinds', async () => {
    postJSON.mockResolvedValue({ tag: 'tryit' });
    setup();
    fireEvent.change(optimizer(), { target: { value: 'schedule_free' } });
    fireEvent.click(createButton());
    await waitFor(() => expect(postJSON).toHaveBeenCalledWith('/api/tasks', {
      workload: 'position_eval',
      tag: 'tryit',
      params: { optimizer: 'schedule_free', lr: 0.0025, batch_size: 256 },
    }));
  });

  it('keeps a hand-edited rate until the arm changes under it', () => {
    setup();
    fireEvent.change(optimizer(), { target: { value: 'wsd' } });
    fireEvent.change(lr(), { target: { value: '0.0003' } });
    expect(lr().value).toBe('0.0003');
    fireEvent.change(optimizer(), { target: { value: 'schedule_free' } });
    expect(lr().value).toBe('0.0025');
  });
});
