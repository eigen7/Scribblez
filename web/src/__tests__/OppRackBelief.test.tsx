import { describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import OppRackBelief from '../components/OppRackBelief';
import { RackBeliefChart } from '../lib/surveyTypes';

const TILES = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ?'];

const RACKS = [
  { leave: 'DOR?', refill: 'AT?' },
  { leave: 'ADRT', refill: 'OO?' },
];

function chart(modelled: boolean): RackBeliefChart {
  return {
    modelled,
    leave_length: modelled ? 4 : 0,
    refill: modelled ? 3 : 7,
    presence: TILES.map((tile, i) => ({
      tile,
      inferred: modelled && tile === '?' ? 0.95 : i % 2 ? 0.5 : 0,
      uniform: tile === '?' ? 0.93 : i % 2 ? 0.5 : 0,
    })),
    racks: RACKS,
    model: 'v1 gen 292',
  };
}

describe('OppRackBelief', () => {
  it('draws a pair of bars per tile and reads one out on hover', () => {
    render(<OppRackBelief chart={chart(true)} />);
    expect(screen.getByText(/kept 4 at their last move, drew 3/)).toBeInTheDocument();
    const columns = document.querySelectorAll('.belief-chart g');
    expect(columns).toHaveLength(27);
    // Zero-height bars draw nothing; the blank's two bars both stand.
    expect(columns[26].querySelectorAll('path[d^="M"]')).toHaveLength(2);
    fireEvent.mouseEnter(columns[26]);
    expect(screen.getByText('?: inferred 95.0%, uniform 93.0%')).toBeInTheDocument();
  });

  it('shows each sample rack kept tiles first', () => {
    render(<OppRackBelief chart={chart(true)} />);
    expect(document.querySelectorAll('.belief-sample')).toHaveLength(2);
    const first = document.querySelector('.belief-sample')!;
    expect([...first.querySelectorAll('.kept')].map((e) => e.textContent).join('')).toBe('DOR?');
    expect([...first.querySelectorAll('.drawn')].map((e) => e.textContent).join('')).toBe('AT?');
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('names its model, and the pools\' own when it stands in for that one', () => {
    const { rerender } = render(<OppRackBelief chart={chart(true)} />);
    expect(screen.getByText('leave inference v1 gen 292')).toBeInTheDocument();
    rerender(<OppRackBelief chart={{ ...chart(true), stand_in_for: 'tilted2 gen 1188' }} />);
    expect(
      screen.getByText('leave inference v1 gen 292, standing in for tilted2 gen 1188 that labeled the pools'),
    ).toBeInTheDocument();
  });

  it('draws inferred bars equal to the uniform ones where the model has nothing to say', () => {
    const flat = chart(false);
    flat.presence = flat.presence.map((t) => ({ ...t, inferred: t.uniform }));
    render(<OppRackBelief chart={flat} />);
    expect(screen.getByText(/no leave inference here, so inferred is uniform: a uniform rack of 7/)).toBeInTheDocument();
    expect(screen.getByText('inferred')).toBeInTheDocument();
    const [inferred, uniform] = document.querySelectorAll('.belief-chart g')[26].querySelectorAll('path[d^="M"]');
    // The same height: only the x offset differs.
    const ys = (p: Element) => p.getAttribute('d')!.match(/V([\d.]+)/)![1];
    expect(ys(inferred)).toBe(ys(uniform));
  });
});
