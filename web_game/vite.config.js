import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    fs: {
      allow: [
        "/Users/dpzmick/programming/othello"
      ]
    }
  },
  plugins: [svelte()],
})
