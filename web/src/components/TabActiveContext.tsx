import { createContext } from 'react';

// Whether the subtree's task-view tab is the visible one. Visited tabs stay
// mounted behind display:none (TaskView) so switching back is instant -- their
// embedded figures survive -- and consumers use this to pause background work
// (useFigureItem stops version-polling) while hidden, catching up on
// re-activation. Defaults to true so components used outside the tab system
// behave as always-visible.
export const TabActiveContext = createContext(true);
