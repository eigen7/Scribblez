/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Ports are injected by the engine (play_game) when it launches this dev server,
// so the browser UI and the C++ WebSocket server always agree. They default to
// vite's 5173 and play_game's 8080 for a standalone `npm run dev`.
const devPort = Number(process.env.VITE_DEV_PORT ?? 5173);
const wsPort = Number(process.env.VITE_WS_PORT ?? 8080);

export default defineConfig({
  plugins: [react()],
  server: {
    port: devPort,
    strictPort: true,
    proxy: {
      '/ws': {
        target: `ws://localhost:${wsPort}`,
        ws: true,
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
