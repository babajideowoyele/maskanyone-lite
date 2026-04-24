import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The backend is `python` service inside docker compose. When running via
// `yarn dev` outside Docker, override with VITE_API_BASE=http://localhost:8000
const target = process.env.VITE_API_BASE || 'http://python:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/mask': target,
      '/platform': target,
    },
  },
});
