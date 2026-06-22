import React from 'react';

interface ScoreBoardProps {
  playerNames: [string, string];
  scores: [number, number];
  bagCount: number;
  opponentRackCount: number;
  yourTurn: boolean;
  gameOver: boolean;
  winner?: number;
  rackCounts?: [number, number];
  editableNames?: boolean;
  maxNameLength?: number;
  onNameChange?: (player: 0 | 1, name: string) => void;
  showTurnIndicator?: boolean;
}

const ScoreBoard: React.FC<ScoreBoardProps> = ({
  playerNames,
  scores,
  bagCount,
  opponentRackCount,
  yourTurn,
  gameOver,
  winner,
  rackCounts,
  editableNames = false,
  maxNameLength = 14,
  onNameChange,
  showTurnIndicator = true,
}) => {
  const handleNameEdit = (player: 0 | 1, value: string) => {
    const trimmed = value.slice(0, maxNameLength);
    onNameChange?.(player, trimmed);
  };

  return (
    <div className="scoreboard">
      <div className={`score-entry${yourTurn && !gameOver ? ' active' : ''}`}>
        {editableNames ? (
          <input
            className="score-name score-name-input"
            value={playerNames[0]}
            maxLength={maxNameLength}
            onChange={(e) => handleNameEdit(0, e.target.value)}
          />
        ) : (
          <span className="score-name">{playerNames[0]}</span>
        )}
        <span className="score-value">{scores[0]}</span>
      </div>
      <div className={`score-entry${!yourTurn && !gameOver ? ' active' : ''}`}>
        {editableNames ? (
          <input
            className="score-name score-name-input"
            value={playerNames[1]}
            maxLength={maxNameLength}
            onChange={(e) => handleNameEdit(1, e.target.value)}
          />
        ) : (
          <span className="score-name">{playerNames[1]}</span>
        )}
        <span className="score-value">{scores[1]}</span>
      </div>
      <div className="game-info">
        <span>Bag: {bagCount} tiles</span>
        <span>
          {rackCounts ? `${rackCounts[0]} / ${rackCounts[1]} rack` : `Opponent rack: ${opponentRackCount}`}
        </span>
      </div>
      {gameOver && (
        <div className="game-over-banner">
          Game Over — {winner === 0 ? 'You win!' : winner === 1 ? 'AI wins!' : 'Tie!'}
        </div>
      )}
      {!gameOver && showTurnIndicator && (
        <div className="turn-indicator">
          {yourTurn ? 'Your turn' : 'AI is thinking...'}
        </div>
      )}
    </div>
  );
};

export default ScoreBoard;
