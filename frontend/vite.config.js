import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  // Keep one project-level environment file. Vite still exposes only VITE_*
  // variables to browser code, so backend credentials remain private.
  envDir: '..',
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/@pipecat-ai/')) return 'pipecat';
          if (id.includes('node_modules/react') || id.includes('node_modules/react-dom')) return 'react';
          return undefined;
        },
      },
    },
  },
  server: {
    host: '0.0.0.0'
  }
})
