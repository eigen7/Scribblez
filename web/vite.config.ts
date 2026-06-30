/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Ports are injected by the engine (play_game) when it launches this dev server,
// so the browser UI and the C++ WebSocket server always agree. They default to
// vite's 5173 and play_game's 8080 for a standalone `npm run dev`.
const devPort = Number(process.env.VITE_DEV_PORT ?? 5173);
const wsPort = Number(process.env.VITE_WS_PORT ?? 8080);
// The React dashboard talks to the Python (Tornado) data API; the engine injects
// this port when it launches the dashboard dev server. Defaults to the API's 8090.
const apiPort = Number(process.env.VITE_API_PORT ?? 8090);

export default defineConfig({
  plugins: [react()],
  // The engine/trainer launches Vite with the terminal inherited, so Vite's
  // default screen-clearing on startup and restart would wipe the host process's
  // output (e.g. a trainer's progress log). Keep the terminal intact.
  clearScreen: false,
  server: {
    // Listen on all interfaces so that when play_game runs inside a Docker
    // container (see run_docker.py), the host's browser can reach the dev
    // server through the forwarded port. Defaults to 127.0.0.1, which is
    // unreachable from outside the container.
    host: true,
    port: devPort,
    strictPort: true,
    proxy: {
      '/ws': {
        target: `ws://localhost:${wsPort}`,
        ws: true,
      },
      '/api': {
        target: `http://localhost:${apiPort}`,
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.ts',
    css: false,
  },
});
