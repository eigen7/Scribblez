import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import AppManual from './AppManual';
import AppBoard from './AppBoard';
import MasterApp from './components/master/MasterApp';
import './index.css';

// The engine injects VITE_TOOL when it launches the dev server, so each tool's
// bare URL serves the right UI. A standalone `npm run dev` has no such env var
// and falls back to a `?tool=` query param (default: play_game's App).
const params = new URLSearchParams(window.location.search);
const tool = import.meta.env.VITE_TOOL ?? params.get('tool');
const Root =
  tool === 'manual' ? AppManual
  : tool === 'board' ? AppBoard
  : tool === 'dashboard' ? MasterApp
  : App;

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
