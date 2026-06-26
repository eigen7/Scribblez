/// <reference types="vite/client" />

interface ImportMetaEnv {
  // Which tool's UI to mount, injected by the engine when it launches the dev
  // server (e.g. "manual"). Absent for a standalone `npm run dev`.
  readonly VITE_TOOL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
