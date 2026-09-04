import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
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
  profiles: {},
  default_profile: '',
};

// The same workload with two recipes: the transformer one sets a larger batch.
const profiled: Workload = {
  ...workload,
  primary_params: ['optimizer', 'batch_size'],
  profiles: { transformer: { batch_size: 512 }, cnn: {} },
  default_profile: 'transformer',
};

const setup = (w: Workload = workload) => {
  render(<NewTagForm workload={w} onCreated={() => {}} />);
  fireEvent.change(screen.getByPlaceholderText('e.g. exp42'), { target: { value: 'tryit' } });
};

const optimizer = () => screen.getByLabelText('optimizer') as HTMLSelectElement;
const lr = () => screen.getByLabelText('lr') as HTMLInputElement;
const batch = () => screen.getByLabelText('batch_size') as HTMLInputElement;
const profileSelect = () => screen.getByLabelText('profile') as HTMLSelectElement;
const advancedToggle = () => screen.getByText(/Advanced:/);
const createButton = () => screen.getByText('Create').closest('button') as HTMLButtonElement;

describe('NewTagForm with a closed parameter', () => {
  beforeEach(() => {
    postJSON.mockReset();
    window.localStorage.clear();
  });

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
  beforeEach(() => {
    postJSON.mockReset();
    window.localStorage.clear();
  });

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

describe('NewTagForm profiles', () => {
  beforeEach(() => {
    postJSON.mockReset();
    window.localStorage.clear();
  });

  it('shows no selector for a workload without profiles, and keeps its plain form across a remount', () => {
    setup();
    expect(screen.queryByLabelText('profile')).toBeNull();
    fireEvent.click(advancedToggle());
    fireEvent.change(lr(), { target: { value: '0.0003' } });
    // a second mount (a reload) starts from the defaults again
    const again = within(render(<NewTagForm workload={workload} onCreated={() => {}} />).container);
    fireEvent.click(again.getByText(/Advanced:/));
    expect((again.getByLabelText('lr') as HTMLInputElement).value).toBe('0');
  });

  it('starts from the default profile and marks the values it sets', () => {
    setup(profiled);
    expect(profileSelect().value).toBe('transformer');
    expect(batch().value).toBe('512');
    expect(screen.getByTitle('set by the transformer profile')).toBeTruthy();
    expect(optimizer().value).toBe('wsd');  // a field the profile is silent on: the schema default
  });

  it('scopes edits to the profile they were made under', () => {
    setup(profiled);
    fireEvent.change(batch(), { target: { value: '384' } });
    expect(screen.getByTitle(/edited; click to reset/)).toBeTruthy();
    fireEvent.change(profileSelect(), { target: { value: 'cnn' } });
    expect(batch().value).toBe('256');  // the cnn recipe, untouched by the transformer edit
    fireEvent.change(profileSelect(), { target: { value: 'transformer' } });
    expect(batch().value).toBe('384');  // the edit survives the round trip
  });

  it('resets an edited field to the profile value, and drops an edit that lands on it', () => {
    setup(profiled);
    fireEvent.change(batch(), { target: { value: '384' } });
    fireEvent.click(screen.getByTitle(/edited; click to reset/));
    expect(batch().value).toBe('512');
    fireEvent.change(batch(), { target: { value: '512' } });  // typing the profile's own value
    expect(screen.queryByTitle(/edited; click to reset/)).toBeNull();
  });

  it('submits the selected profile with its resolved values', async () => {
    postJSON.mockResolvedValue({ tag: 'tryit' });
    setup(profiled);
    fireEvent.change(profileSelect(), { target: { value: 'cnn' } });
    fireEvent.change(batch(), { target: { value: '128' } });
    fireEvent.click(createButton());
    await waitFor(() => expect(postJSON).toHaveBeenCalledWith('/api/tasks', {
      workload: 'position_eval',
      tag: 'tryit',
      params: { optimizer: 'wsd', lr: 0, batch_size: 128 },
      profile: 'cnn',
    }));
  });

  it('keeps drafts and the selected profile across a remount', () => {
    setup(profiled);
    fireEvent.change(profileSelect(), { target: { value: 'cnn' } });
    fireEvent.change(batch(), { target: { value: '64' } });
    // a second mount (a reload) picks the drafts back up from storage
    const again = within(render(<NewTagForm workload={profiled} onCreated={() => {}} />).container);
    expect((again.getByLabelText('profile') as HTMLSelectElement).value).toBe('cnn');
    expect((again.getByLabelText('batch_size') as HTMLInputElement).value).toBe('64');
  });
});
