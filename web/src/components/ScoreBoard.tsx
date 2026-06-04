import React from 'react';

interface ScoreBoardProps {
  playerNames: [string, string];
  scores: [number, number];
  bagCount: number;
  opponentRackCount: number;
  yourTurn: boolean;
  gameOver: boolean;
  winner?: number;
}

const ScoreBoard: React.FC<ScoreBoardProps> = ({
  playerNames,
  scores,
  bagCount,
  opponentRackCount,
  yourTurn,
  gameOver,
  winner,
}) => {
  return (
    <div className="scoreboard">
      <div className={`score-entry${yourTurn && !gameOver ? ' active' : ''}`}>
        <span className="score-name">{playerNames[0]}</span>
        <span className="score-value">{scores[0]}</span>
      </div>
      <div className={`score-entry${!yourTurn && !gameOver ? ' active' : ''}`}>
        <span className="score-name">{playerNames[1]}</span>
        <span className="score-value">{scores[1]}</span>
      </div>
      <div className="game-info">
        <span>Bag: {bagCount} tiles</span>
        <span>Opponent rack: {opponentRackCount}</span>
      </div>
      {gameOver && (
        <div className="game-over-banner">
          Game Over — {winner === 0 ? 'You win!' : winner === 1 ? 'AI wins!' : 'Tie!'}
        </div>
      )}
      {!gameOver && (
        <div className="turn-indicator">
          {yourTurn ? 'Your turn' : 'AI is thinking...'}
        </div>
      )}
    </div>
  );
};

export default ScoreBoard;
