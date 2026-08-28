// Offline board-image driver: rasterizes engine-dumped `manual_state` JSON to
// PNGs by loading the `?tool=render` harness in a headless browser and reading
// back the PNG it captures via the shared Export-PNG path.
//
// Usage:  node web/scripts/render_boards.mjs <manifest.json> [--port 5199]
//
// Manifest shape:
//   { "states": [ { "state": "states/ply_21.json",
//                   "out": "images/critical.png",
//                   "caption": "Nigel to play, up 53" }, ... ] }
// Paths in the manifest are resolved relative to the manifest file's directory.
// The Vite dev server for web/ is started and stopped by this script; no engine
// or WebSocket is involved (the state is pre-dumped).

import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import http from 'node:http';

const WEB_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..');

// Standard English Scrabble tile distribution (blanks tracked separately and
// never surface as unseen letters, so they're omitted here).
const TILE_DISTRIBUTION = {
  A: 9, B: 2, C: 2, D: 4, E: 12, F: 2, G: 3, H: 2, I: 9, J: 1, K: 1, L: 4, M: 2,
  N: 6, O: 8, P: 2, Q: 1, R: 6, S: 4, T: 6, U: 4, V: 2, W: 2, X: 1, Y: 2, Z: 1,
};

// The tiles unseen from `seat`'s point of view (the bag plus the opponent's
// rack): the full distribution minus the tiles on the board and in `seat`'s own
// rack. Board blanks appear lowercase and consume no letter, so they're skipped.
function computeUnseen(state, seat) {
  const counts = { ...TILE_DISTRIBUTION };
  for (const row of state.board) {
    for (const cell of row) {
      if (cell && cell === cell.toUpperCase() && /[A-Z]/.test(cell)) counts[cell]--;
    }
  }
  for (const slot of state.racks[seat]) {
    if (slot.present && slot.known && counts[slot.letter] !== undefined) counts[slot.letter]--;
  }
  const unseen = [];
  for (const letter of Object.keys(counts).sort()) {
    for (let i = 0; i < counts[letter]; i++) unseen.push(letter);
  }
  return unseen;
}

function arg(flag, fallback) {
  const i = process.argv.indexOf(flag);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

function waitForServer(port, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolvePromise, reject) => {
    const tick = () => {
      const req = http.get({ host: 'localhost', port, path: '/' }, (res) => {
        res.resume();
        resolvePromise();
      });
      req.on('error', () => {
        if (Date.now() > deadline) reject(new Error('Vite dev server did not start'));
        else setTimeout(tick, 300);
      });
    };
    tick();
  });
}

async function main() {
  const manifestPath = process.argv[2];
  if (!manifestPath || manifestPath.startsWith('--')) {
    throw new Error('usage: node render_boards.mjs <manifest.json> [--port N]');
  }
  const port = Number(arg('--port', '5199'));
  const manifestDir = dirname(resolve(manifestPath));
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));

  // VITE_TOOL must be absent (not empty) so the app's `VITE_TOOL ?? ?tool=`
  // fallback honors our `?tool=render` query param.
  const viteEnv = { ...process.env };
  delete viteEnv.VITE_TOOL;
  const vite = spawn('npm', ['run', 'dev', '--', '--port', String(port), '--strictPort'], {
    cwd: WEB_DIR,
    env: viteEnv,
    stdio: ['ignore', 'inherit', 'inherit'],
  });

  let browser;
  try {
    await waitForServer(port);
    browser = await chromium.launch();
    for (const item of manifest.states) {
      const state = JSON.parse(readFileSync(resolve(manifestDir, item.state), 'utf8'));
      if (item.caption) state.caption = item.caption;
      // "unseenFrom": <seat> adds an unseen-tiles strip from that seat's POV.
      if (item.unseenFrom !== undefined) {
        state.unseen = computeUnseen(state, item.unseenFrom);
        state.unseenLabel = state.player_names[item.unseenFrom];
      }
      // "highlights"/"legend" annotate squares (no tiles placed) and explain them.
      if (item.highlights) state.highlights = item.highlights;
      if (item.legend) state.legend = item.legend;
      // Blank a seat's rack to 7 unseen "?" tiles (a player-POV view): e.g.
      // "hideRacks": [0] hides Mike's rack in a Nigel-POV figure.
      for (const seat of item.hideRacks ?? []) {
        state.racks[seat] = Array.from({ length: 7 }, () => ({
          letter: '', score: 0, known: false, present: true, drawn: false,
        }));
      }

      const context = await browser.newContext({ deviceScaleFactor: 2 });
      await context.addInitScript((s) => {
        window.__RENDER_STATE__ = s;
      }, state);
      const page = await context.newPage();
      page.on('pageerror', (e) => console.error(`[page error] ${e.message}`));
      page.on('console', (m) => {
        if (m.type() === 'error') console.error(`[console] ${m.text()}`);
      });
      await page.goto(`http://localhost:${port}/?tool=render`, { waitUntil: 'load' });
      await page.waitForFunction(
        () => window.__RENDER_PNG__ !== undefined || window.__RENDER_ERROR__ !== undefined,
        { timeout: 30000 },
      );
      const err = await page.evaluate(() => window.__RENDER_ERROR__);
      if (err) throw new Error(`render failed for ${item.state}: ${err}`);
      const dataUrl = await page.evaluate(() => window.__RENDER_PNG__);

      const outPath = resolve(manifestDir, item.out);
      mkdirSync(dirname(outPath), { recursive: true });
      writeFileSync(outPath, Buffer.from(dataUrl.split(',')[1], 'base64'));
      console.log(`wrote ${outPath}`);
      await context.close();
    }
  } finally {
    if (browser) await browser.close();
    vite.kill('SIGTERM');
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
