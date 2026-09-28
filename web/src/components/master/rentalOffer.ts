import { getJSON } from '../../lib/api';

// The provider, the account it rents as, and its catalog (GET
// /api/cloud/rental_offer), as the task view's rent form and the pool's
// capacity form show it.

export type MachineType = {
  id: string; vcpus: number; gpu_count: number; gpu: string; arch: string; cost_per_hr: number;
};
export type RentalOffer = {
  provider: string; account: string; types: MachineType[];
  spot_prices: Record<string, number>;  // current spot rate by type id, where readable
};

// Fetched once per page load for every task's rent form; a failed fetch (no
// aws credentials yet) is retried by the next caller rather than cached.
let offerPromise: Promise<RentalOffer> | null = null;
export function loadRentalOffer(): Promise<RentalOffer> {
  if (!offerPromise) {
    offerPromise = getJSON('/api/cloud/rental_offer').catch((e) => {
      offerPromise = null;
      throw e;
    });
  }
  return offerPromise;
}

// A type's rate under the chosen market; a spot rate the account may not read
// (or the region has none of) is unknown, not the list price.
export function rateText(offer: RentalOffer, typeId: string, spot: boolean): string {
  const rate = spot ? offer.spot_prices[typeId] : offer.types.find((t) => t.id === typeId)?.cost_per_hr;
  return rate != null ? `$${rate.toFixed(3)}/hr` : 'rate unknown';
}
