// The survey viewer's navigation state, kept in the URL's hash so a refresh (or
// a copied link) reopens the same position under the same bag filter:
// "#pos=<position name>&bag=<min>-<max>", either bound of the range optional.
import { SurveyPosition } from './surveyTypes';

export interface BagFilter {
  min: number | null;
  max: number | null;
}

export interface NavState {
  pos: string | null; // a position's name
  bag: BagFilter;
}

const bound = (s: string | undefined): number | null =>
  s === undefined || s === '' || Number.isNaN(Number(s)) ? null : Number(s);

export function parseNav(hash: string): NavState {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const [min, max] = (params.get('bag') ?? '').split('-');
  return { pos: params.get('pos'), bag: { min: bound(min), max: bound(max) } };
}

export function formatNav(nav: NavState): string {
  const params = new URLSearchParams();
  if (nav.pos !== null) params.set('pos', nav.pos);
  if (nav.bag.min !== null || nav.bag.max !== null)
    params.set('bag', `${nav.bag.min ?? ''}-${nav.bag.max ?? ''}`);
  return `#${params.toString()}`;
}

export const passesBag = (p: SurveyPosition, bag: BagFilter) =>
  (bag.min === null || p.bag_size >= bag.min) && (bag.max === null || p.bag_size <= bag.max);
