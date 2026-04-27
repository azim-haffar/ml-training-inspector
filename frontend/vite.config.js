import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    watch: {
      // Polling is needed for hot reload to work inside Docker on Windows/Mac
      usePolling: true,
    },
  },
})
