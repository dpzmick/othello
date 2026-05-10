import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// https://vitejs.dev/config/
export default defineConfig({
  base: '/static/othello2023/',
  server: {
    fs: {
      // The wasm bundle lives at ../emcc-build/web_game/, so the dev server
      // needs read access to the project root.
      allow: [path.resolve(__dirname, '..')],
    },
  },
  plugins: [svelte()],
})
