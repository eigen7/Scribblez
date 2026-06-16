// Render board positions to PNG images that match the live web UI.
//
// Reads a JSON file holding an array of { name, state } entries, where `state`
// is the web GameState (as emitted by the C++ scribblez_dump_position_json
// FFI). Each position is rendered with the real <ScoreBoard>/<Board>/<Rack>
// components and the app's index.css, then screenshotted via headless Chromium
// to <outDir>/<name>.png.
//
// Usage: tsx tools/render_positions.tsx <positions.json> <outDir>

import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { chromium } from 'playwright';

import Board from '../src/components/Board';
import Rack from '../src/components/Rack';
import ScoreBoard from '../src/components/ScoreBoard';
import { GameState } from '../src/types';

interface Position {
  name: string;
  state: GameState;
}

const here = dirname(fileURLToPath(import.meta.url));
const css = readFileSync(join(here, '../src/index.css'), 'utf-8');

const [, , inputPath, outDir] = process.argv;
if (!inputPath || !outDir) {
  console.error('Usage: tsx tools/render_positions.tsx <positions.json> <outDir>');
  process.exit(2);
}

const positions: Position[] = JSON.parse(readFileSync(inputPath, 'utf-8'));
const noop = () => {};

function layout(state: GameState): React.ReactElement {
  return (
    <div className="game-layout">
      <div className="board-section">
        <Board
          board={state.board}
          bonuses={state.bonuses}
          candidateTiles={[]}
          tileScores={state.tile_scores ?? {}}
          cursorRow={null}
          cursorCol={null}
          cursorDir={null}
          interactive={false}
          onCellClick={noop}
          onCellDrop={noop}
        />
        <Rack tiles={state.rack} usedIndices={new Set()} label="Leave (POV rack)" interactive={false} />
      </div>
      <div className="sidebar">
        <ScoreBoard
          playerNames={state.player_names}
          scores={state.scores}
          bagCount={state.bag_count}
          opponentRackCount={state.opponent_rack_count}
          yourTurn={state.your_turn}
          gameOver={false}
        />
      </div>
    </div>
  );
}

mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 2 });

for (const { name, state } of positions) {
  const body = renderToStaticMarkup(layout(state));
  const html =
    '<!doctype html><html><head><meta charset="utf-8"><style>' +
    css +
    '\nbody{display:inline-block;padding:16px;}</style></head><body>' +
    body +
    '</body></html>';
  await page.setContent(html, { waitUntil: 'networkidle' });
  const el = await page.$('.game-layout');
  if (!el) throw new Error(`render failed for ${name}: .game-layout not found`);
  await el.screenshot({ path: join(outDir, `${name}.png`) });
  console.log(`rendered ${name}.png`);
}

await browser.close();
