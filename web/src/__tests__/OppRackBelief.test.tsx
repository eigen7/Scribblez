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

  it('names its model', () => {
    render(<OppRackBelief chart={chart(true)} />);
    expect(screen.getByText('leave inference v1 gen 292')).toBeInTheDocument();
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

  it("draws the kept leave's value as a normal curve per belief and reads them out", () => {
    const value = { inferred: { mean: 10, sd: 5 }, uniform: { mean: 10, sd: 10 } };
    render(<OppRackBelief chart={{ ...chart(true), leave_value: value }} />);
    expect(document.querySelector('.leave-value-inferred')).not.toBeNull();
    expect(document.querySelector('.leave-value-uniform')!.getAttribute('stroke-dasharray')).toBe('5 3');
    expect(screen.getByText('inferred +10.0 ± 5.0 · uniform +10.0 ± 10.0')).toBeInTheDocument();
    // The middle of the axis is both means: half of each curve lies below.
    fireEvent.mouseMove(document.querySelector('.leave-value-chart')!, { clientX: 255 });
    expect(screen.getByText('worth ≤ +10.0: inferred 50%, uniform 50%')).toBeInTheDocument();
  });

  it('has no value curves where the opponent kept no leave', () => {
    render(<OppRackBelief chart={{ ...chart(false), leave_value: null }} />);
    expect(document.querySelector('.leave-value')).toBeNull();
    expect(screen.getByText('No kept leave is modelled here, so no leave value to chart.')).toBeInTheDocument();
  });
});
