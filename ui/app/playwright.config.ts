// Playwright E2E config (P6 task 7, P6_SPEC §4 "End-to-end"). Two app servers:
// up-mode (scripted graph model + seeded live graph, port 8765) and off-mode
// (degradation + replay, no DB, port 8766). Servers run from the repo root so
// relative artifact paths (ui/app/dist, evals/*) resolve. The UI must be built
// first (`make ui-build`) — the servers serve dist/.
import { defineConfig } from "@playwright/test";

const REPO_ROOT = "../..";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false, // one shared scripted server; keep turn order deterministic
  workers: 1,
  retries: 0,
  use: {
    baseURL: "http://127.0.0.1:8765",
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: "uv run python tests/e2e/e2e_server.py",
      cwd: REPO_ROOT,
      port: 8765,
      env: { E2E_MODE: "up", E2E_PORT: "8765" },
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: "uv run python tests/e2e/e2e_server.py",
      cwd: REPO_ROOT,
      port: 8766,
      env: { E2E_MODE: "off", E2E_PORT: "8766" },
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
