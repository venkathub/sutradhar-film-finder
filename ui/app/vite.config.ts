/// <reference types="vitest/config" />
// Vite build + Vitest browser-mode config (P6 task 2, DEC-P6-1).
// - build: pure static assets to dist/ (served same-origin by FastAPI / the app image);
// - dev: /api proxied to the local FastAPI so `make ui-dev` talks to `make api-up`;
// - test: Vitest 4 Browser Mode via the Playwright provider (real chromium, headless) —
//   the same Playwright install the task-7 E2E tier reuses.
import react from "@vitejs/plugin-react";
import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": `http://localhost:${process.env.API_PORT ?? "8080"}`,
    },
  },
  test: {
    browser: {
      enabled: true,
      headless: true,
      provider: playwright(),
      instances: [{ browser: "chromium" }],
    },
  },
});
