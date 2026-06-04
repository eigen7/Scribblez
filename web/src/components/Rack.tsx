import React from 'react';
import { TileInfo } from '../types';

interface RackProps {
  tiles: TileInfo[];
  usedIndices: Set<number>; // indices of tiles that have been placed on board
  label: string;
  interactive: boolean;
}

const Rack: React.FC<RackProps> = ({ tiles, usedIndices, label, interactive }) => {
  const handleDragStart = (e: React.DragEvent, tile: TileInfo, index: number) => {
    if (!interactive || usedIndices.has(index)) {
      e.preventDefault();
      return;
    }
    const isBlank = tile.letter === '?';
    e.dataTransfer.setData('text/plain', JSON.stringify({
      letter: isBlank ? '?' : tile.letter,
      isBlank,
      rackIndex: index,
    }));
    e.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="rack">
      <div className="rack-label">{label}</div>
      <div className="rack-tiles">
        {tiles.map((t, i) => (
          <div
            key={i}
            className={`rack-tile${usedIndices.has(i) ? ' used' : ''}${interactive && !usedIndices.has(i) ? ' draggable-tile' : ''}`}
            draggable={interactive && !usedIndices.has(i)}
            onDragStart={(e) => handleDragStart(e, t, i)}
          >
            <span className="rack-tile-letter">{t.letter === '?' ? '' : t.letter}</span>
            <span className="rack-tile-score">{t.score}</span>
          </div>
        ))}
        {/* Empty slots */}
        {Array.from({ length: 7 - tiles.length }, (_, i) => (
          <div key={`empty-${i}`} className="rack-tile empty" />
        ))}
      </div>
    </div>
  );
};

export default Rack;
