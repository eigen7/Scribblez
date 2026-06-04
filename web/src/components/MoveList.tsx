import React, { useState } from 'react';
import { MoveOption } from '../types';

interface MoveListProps {
  moves: MoveOption[];
  selectedIndex: number | null;
  onPreview: (index: number) => void;
  disabled: boolean;
}

const MoveList: React.FC<MoveListProps> = ({ moves, selectedIndex, onPreview, disabled }) => {
  const [filter, setFilter] = useState('');

  const filtered = filter
    ? moves.filter((m) => m.text.toLowerCase().includes(filter.toLowerCase()))
    : moves;

  // Show highest-scoring moves first. Copy before sorting so we never mutate
  // the moves prop (and the engine-assigned move.index stays intact).
  const sorted = [...filtered].sort((a, b) => b.score - a.score);

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
        {sorted.map((m) => (
          <button
            key={m.index}
            className={`move-item${selectedIndex === m.index ? ' selected' : ''}`}
            onClick={() => onPreview(m.index)}
            disabled={disabled}
          >
            <span className="move-text">{m.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default MoveList;
