import { TileInfo } from './types';

// One slot of a manual-tool rack, as sent by the engine's `manual_state`. Shared
// by the live manual UI (AppManual) and the offline render harness (RenderBoard).
export interface ManualRackSlot {
  letter: string;
  score: number;
  known: boolean;
  // Whether the player drew this tile after their own last move. Their leave
  // and their replenishment are shaded apart, so a reviewed position shows at a
  // glance which of the tiles they hold their last play accounts for.
  drawn: boolean;
  // Whether a tile occupies this slot at all. A present-but-unknown slot
  // renders as a "?" tile (selectable for exchange); an absent slot renders as
  // empty space.
  present: boolean;
}

// Map an engine rack slot to the TileInfo the <Rack> component draws.
export function rackSlotToTile(slot: ManualRackSlot): TileInfo {
  if (!slot.present) return { letter: '', score: 0, isAbsent: true };
  if (!slot.known) return { letter: '?', score: 0, isUnknown: true };
  if (slot.letter === '?') return { letter: '', score: 0, isBlank: true, isDrawn: slot.drawn };
  return { letter: slot.letter, score: slot.score, isDrawn: slot.drawn };
}
