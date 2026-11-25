import { ReactNode } from 'react';
import ControlsTab from './components/ControlsTab';
import InfoTab from './components/InfoTab';
import LeafEvalPositions from './components/LeafEvalPositions';
import PositionEvalAnalysis from './components/PositionEvalAnalysis';
import { FigureTab, LossTab } from './components/TrainingTabs';

// The client-side workload registry: extra task-view tabs per workload, keyed
// by the registry name the API reports. Generic tabs (Overview, Stats) come
// from TaskView itself; everything here is a workload-specific React
// component. The `task` prop of the training components is the workload name
// -- the data API's task slug and the workload registry key are the same.

export type WorkloadTab = {
  name: string;
  render: (workload: string, tag: string) => ReactNode;
};

const figureTab = (
  name: string, figure: string, versionKey: string, emptyText: string,
): WorkloadTab => ({
  name,
  render: (w, t) => (
    <FigureTab task={w} tag={t} figure={figure} versionKey={versionKey} emptyText={emptyText} />
  ),
});

const trainingTab = (figure: string, versionKey: string, emptyText: string): WorkloadTab =>
  figureTab('Training', figure, versionKey, emptyText);

export const WORKLOAD_TABS: Record<string, WorkloadTab[]> = {
  position_eval: [
    { name: 'Loss', render: (w, t) => <LossTab task={w} tag={t} /> },
    { name: 'Positions', render: (w, t) => <PositionEvalAnalysis task={w} tag={t} /> },
    figureTab('Match', 'match_eval', 'match_eval', 'No match results yet.'),
    trainingTab('training_metrics', 'metrics', 'No per-epoch training metrics yet.'),
    { name: 'Controls', render: (w, t) => <ControlsTab task={w} tag={t} /> },
    { name: 'Info', render: (w, t) => <InfoTab task={w} tag={t} /> },
  ],
  match_arms: [
    figureTab('Arms', 'match_arms', 'match_arm', 'No arm results yet.'),
    { name: 'Info', render: (w, t) => <InfoTab task={w} tag={t} /> },
  ],
  leave_inference: [
    { name: 'Loss', render: (w, t) => <LossTab task={w} tag={t} /> },
    trainingTab('leave_inference_metrics', 'metrics', 'No per-generation metrics yet.'),
    { name: 'Controls', render: (w, t) => <ControlsTab task={w} tag={t} /> },
    { name: 'Info', render: (w, t) => <InfoTab task={w} tag={t} /> },
  ],
  deep_bot_leaf_eval: [
    { name: 'Loss', render: (w, t) => <LossTab task={w} tag={t} /> },
    trainingTab('leaf_eval_metrics', 'metrics', 'No per-generation metrics yet.'),
    figureTab('Match', 'match_eval', 'match_eval', 'No match results yet.'),
    { name: 'Positions', render: (w, t) => <LeafEvalPositions task={w} tag={t} /> },
    { name: 'Controls', render: (w, t) => <ControlsTab task={w} tag={t} /> },
    { name: 'Info', render: (w, t) => <InfoTab task={w} tag={t} /> },
  ],
  leaf_bot_generation: [
    { name: 'Loss', render: (w, t) => <LossTab task={w} tag={t} /> },
    trainingTab('leaf_bot_generation_metrics', 'metrics', 'No generation scored yet.'),
    figureTab('Matches', 'release_matches', 'release_match', 'No match results yet.'),
    { name: 'Positions', render: (w, t) => <LeafEvalPositions task={w} tag={t} /> },
    { name: 'Controls', render: (w, t) => <ControlsTab task={w} tag={t} /> },
    { name: 'Info', render: (w, t) => <InfoTab task={w} tag={t} /> },
  ],
};
