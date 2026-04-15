import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Listen on LAN so you can open the dev app from a phone (same Wi‑Fi).
    host: true,
    proxy: {
      // Browser calls /api/... on the Vite host; Vite forwards to FastAPI on this machine.
      // Fixes "Load failed" on mobile when VITE_API_BASE_URL was localhost (phone ≠ your PC).
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  preview: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
