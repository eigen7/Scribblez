import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { NewTagForm, type Workload } from '../components/master/MasterApp';

// The new-tag form's two schema-driven behaviours: a closed parameter renders
// as a selector over its value set, and the workload's up-front parameters are
// the only ones shown until the Advanced section is expanded.

const postJSON = vi.fn();
vi.mock('../lib/api', () => ({ getJSON: vi.fn(), postJSON: (...a: unknown[]) => postJSON(...a) }));
vi.mock('../components/master/TaskView', () => ({ default: () => null }));

const workload: Workload = {
  name: 'position_eval',
  title: 'Train position evaluation',
  roles: [],
  primary_params: ['optimizer'],
  params: [
    { name: 'optimizer', kind: 'str', default: 'wsd', help: 'the arm', choices: ['wsd', 'schedule_free'] },
    { name: 'lr', kind: 'float', default: 0, help: "0 = the arm's own default", choices: null },
    { name: 'batch_size', kind: 'int', default: 256, help: 'batch', choices: null },
  ],
};

const setup = (w: Workload = workload) => {
  render(<NewTagForm workload={w} onCreated={() => {}} />);
  fireEvent.change(screen.getByPlaceholderText('e.g. exp42'), { target: { value: 'tryit' } });
};

const optimizer = () => screen.getByLabelText('optimizer') as HTMLSelectElement;
const lr = () => screen.getByLabelText('lr') as HTMLInputElement;
const advancedToggle = () => screen.getByText(/Advanced:/);
const createButton = () => screen.getByText('Create').closest('button') as HTMLButtonElement;

describe('NewTagForm with a closed parameter', () => {
  beforeEach(() => postJSON.mockReset());

  it('renders it as a selector over the schema set, at its default', () => {
    setup();
    expect(optimizer().value).toBe('wsd');
    expect([...optimizer().options].map((o) => o.value)).toEqual(['wsd', 'schedule_free']);
  });

  it('submits the values coerced to their kinds', async () => {
    postJSON.mockResolvedValue({ tag: 'tryit' });
    setup();
    fireEvent.change(optimizer(), { target: { value: 'schedule_free' } });
    fireEvent.click(advancedToggle());
    fireEvent.change(lr(), { target: { value: '0.0003' } });
    fireEvent.click(createButton());
    await waitFor(() => expect(postJSON).toHaveBeenCalledWith('/api/tasks', {
      workload: 'position_eval',
      tag: 'tryit',
      params: { optimizer: 'schedule_free', lr: 0.0003, batch_size: 256 },
    }));
  });
});

describe('NewTagForm advanced section', () => {
  beforeEach(() => postJSON.mockReset());

  it('hides the non-primary fields until it is expanded', () => {
    setup();
    expect(screen.queryByLabelText('lr')).toBeNull();
    expect(advancedToggle().textContent).toContain('2 more settings');
    fireEvent.click(advancedToggle());
    expect(lr().value).toBe('0');  // the trainer resolves 0 to the arm's rate
  });

  it('shows every field up front when the workload names no primary ones', () => {
    setup({ ...workload, primary_params: [] });
    expect(lr().value).toBe('0');
    expect(screen.queryByText(/Advanced:/)).toBeNull();
  });

  it('stays open while a field inside it is invalid', () => {
    setup();
    fireEvent.click(advancedToggle());
    fireEvent.change(lr(), { target: { value: 'oops' } });
    expect(createButton().disabled).toBe(true);
    fireEvent.click(advancedToggle());  // a collapse that would hide the culprit
    expect(lr()).toBeTruthy();
  });
});
