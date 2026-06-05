import React, { useState } from 'react';
import { MoveOption } from '../types';

interface MoveListProps {
  moves: MoveOption[];
  selectedIndex: number | null;
  onPreview: (index: number) => void;
  disabled: boolean;
}

type SortColumn = 'text' | 'score' | 'equity';
type SortDir = 'asc' | 'desc';

// Sort moves by the chosen column. Equity-null rows are sorted to the bottom
// regardless of direction so an unevaluated row never displaces an evaluated
// one in the visible top of the list.
function sortMoves(moves: MoveOption[], col: SortColumn, dir: SortDir): MoveOption[] {
  const sign = dir === 'asc' ? 1 : -1;
  const copy = [...moves];
  copy.sort((a, b) => {
    if (col === 'text') {
      return a.text.localeCompare(b.text) * sign;
    }
    if (col === 'score') {
      return (a.score - b.score) * sign;
    }
    // equity
    const ae = a.equity ?? null;
    const be = b.equity ?? null;
    if (ae === null && be === null) return 0;
    if (ae === null) return 1; // nulls last
    if (be === null) return -1;
    return (ae - be) * sign;
  });
  return copy;
}

// "▲"/"▼" indicator next to the active sort column. Always rendered (even on
// inactive columns) as a fixed-width span so the surrounding header text
// doesn't shift sideways when the sort column changes.
function SortArrow({ active, dir }: { active: boolean; dir: SortDir }) {
  const glyph = !active ? '\u00A0' : dir === 'asc' ? '▲' : '▼';
  return <span className="sort-arrow">{glyph}</span>;
}

// Drop the trailing score token from a move's display text -- e.g. turn
// "8H WAREZ 54" into "8H WAREZ" -- so we don't duplicate the Pts column.
// Non-play moves like "(pass)" or "(exch ABC)" are returned unchanged.
function moveDisplay(text: string): string {
  const m = text.match(/^(.*\S)\s+-?\d+$/);
  return m ? m[1] : text;
}

const MoveList: React.FC<MoveListProps> = ({ moves, selectedIndex, onPreview, disabled }) => {
  const [filter, setFilter] = useState('');
  // Default: best (highest) equity first. Clicking a column header sorts by
  // that column; clicking the same header again flips direction.
  const [sortCol, setSortCol] = useState<SortColumn>('equity');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const filtered = filter
    ? moves.filter((m) => m.text.toLowerCase().includes(filter.toLowerCase()))
    : moves;
  const sorted = sortMoves(filtered, sortCol, sortDir);

  const onHeaderClick = (col: SortColumn) => {
    if (col === sortCol) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortCol(col);
      // Numeric columns default to descending (best first); the move-text
      // column defaults to ascending (A→Z reads more naturally).
      setSortDir(col === 'text' ? 'asc' : 'desc');
    }
  };

  return (
    <div className="move-list">
      <div className="move-list-header">
        <h3>Legal Moves ({moves.length})</h3>
        <input
          type="text"
          placeholder="Filter moves..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          disabled={disabled}
        />
      </div>
      <div className="move-list-scroll">
        <table className="move-table">
          <thead>
            <tr>
              <th className="move-col-text sortable" onClick={() => onHeaderClick('text')}>
                Move<SortArrow active={sortCol === 'text'} dir={sortDir} />
              </th>
              <th className="move-col-num sortable" onClick={() => onHeaderClick('score')}>
                Pts<SortArrow active={sortCol === 'score'} dir={sortDir} />
              </th>
              <th className="move-col-num sortable" onClick={() => onHeaderClick('equity')}>
                Equity<SortArrow active={sortCol === 'equity'} dir={sortDir} />
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m) => {
              const selected = selectedIndex === m.index;
              return (
                <tr
                  key={m.index}
                  className={`move-row${selected ? ' selected' : ''}${disabled ? ' disabled' : ''}`}
                  onClick={() => {
                    if (!disabled) onPreview(m.index);
                  }}
                >
                  <td className="move-col-text">{moveDisplay(m.text)}</td>
                  <td className="move-col-num">{m.score}</td>
                  <td className="move-col-num">
                    {m.equity === null || m.equity === undefined ? '' : m.equity.toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MoveList;
