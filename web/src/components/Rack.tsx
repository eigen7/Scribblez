import React from 'react';
import { TileInfo } from '../types';

interface RackProps {
  tiles: TileInfo[];
  usedIndices: Set<number>; // indices of tiles that have been placed on board
  label: string;
  interactive: boolean;
  // Exchange-mode props: when `exchangeMode` is true, clicking a rack tile
  // toggles it in/out of `exchangeSelected` via `onTileClick`. Dragging is
  // suppressed in exchange mode so the two interaction modes don't collide.
  exchangeMode?: boolean;
  exchangeSelected?: Set<number>;
  onTileClick?: (index: number) => void;
}

const Rack: React.FC<RackProps> = ({
  tiles,
  usedIndices,
  label,
  interactive,
  exchangeMode = false,
  exchangeSelected,
  onTileClick,
}) => {
  const handleDragStart = (e: React.DragEvent, tile: TileInfo, index: number) => {
    if (!interactive || usedIndices.has(index) || exchangeMode) {
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
        {tiles.map((t, i) => {
          const used = usedIndices.has(i);
          const selectedForExchange = exchangeMode && (exchangeSelected?.has(i) ?? false);
          const clickable = interactive && exchangeMode && !used;
          const draggable = interactive && !exchangeMode && !used;
          const classes = [
            'rack-tile',
            used ? 'used' : '',
            draggable ? 'draggable-tile' : '',
            clickable ? 'clickable-tile' : '',
            selectedForExchange ? 'selected-for-exchange' : '',
          ]
            .filter(Boolean)
            .join(' ');
          return (
            <div
              key={i}
              className={classes}
              draggable={draggable}
              onDragStart={(e) => handleDragStart(e, t, i)}
              onClick={clickable ? () => onTileClick?.(i) : undefined}
            >
              <span className="rack-tile-letter">{t.letter === '?' ? '' : t.letter}</span>
              <span className="rack-tile-score">{t.score}</span>
            </div>
          );
        })}
        {/* Empty slots */}
        {Array.from({ length: 7 - tiles.length }, (_, i) => (
          <div key={`empty-${i}`} className="rack-tile empty" />
        ))}
      </div>
    </div>
  );
};

export default Rack;
