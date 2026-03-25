import { defineConfig } from "vite";

export default defineConfig({
  server: {
    proxy: {
      "/api": "http://127.0.0.1:5111",
      "/partials": "http://127.0.0.1:5111",
    },
  },
});
