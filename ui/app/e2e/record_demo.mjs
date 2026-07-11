// Demo-video recorder (P6 task 11 follow-up, Q2/DEC-P6-3).
//
// ONE continuous Playwright take (context.recordVideo -> WebM), three scenes:
//   1. Zero-GPU story  — offline notice + replay browser (GS-08a: cards, citations, trace)
//      against the off-mode server (:8766).
//   2. Live story      — real turns through the live app (:8080, on-demand GPU up):
//      Papanasam -> full version set -> citation -> decoy -> backtrack -> trace view.
//   3. STOP on camera  — the recorder itself runs `make gpu-stop`; the UI flips to
//      "offline by design" on the next status poll. The teardown IS the closing shot.
//
// Run (see docs/RUNBOOK.md): node e2e/record_demo.mjs
// Preconditions: off-mode e2e server on :8766, live demo app on :8080 (GPU window up).
import { chromium } from "playwright";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const run = promisify(execFile);
const OFF = "http://127.0.0.1:8766";
const LIVE = "http://localhost:8080";
const REPO_ROOT = new URL("../../..", import.meta.url).pathname;

const pace = (ms) => new Promise((r) => setTimeout(r, ms));

async function ask(page, text) {
  const input = page.getByLabel("your message");
  await input.click();
  await input.pressSequentially(text, { delay: 35 }); // visible typing, demo pacing
  await pace(400);
  await page.getByRole("button", { name: "Send" }).click();
}

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1280, height: 720 },
  recordVideo: { dir: "demo-video", size: { width: 1280, height: 720 } },
});
const page = await context.newPage();

// --- Scene 1: the zero-GPU story (offline by design + replay browser) ---
await page.goto(OFF + "/");
await page.getByTestId("offline-notice").waitFor();
await pace(3500); // read the offline notice
await page.getByRole("button", { name: "GS-08a", exact: true }).click();
await page.getByTestId("replay-transcript").waitFor();
await pace(2500);
// The recorded version set: scroll to the cards, open citations + the trace view.
const turn2 = page.getByTestId("replay-transcript").getByTestId("turn").nth(1);
await turn2.scrollIntoViewIfNeeded();
await pace(2500);
await turn2.getByTestId("citations").first().locator("summary").click();
await pace(2500);
await turn2.getByTestId("trace-view").locator("summary").click();
await pace(3500);
await page.getByTestId("turn").nth(2).scrollIntoViewIfNeeded();
await pace(2500);

// --- Scene 2: the live story (GPU window up) ---
await page.goto(LIVE + "/");
await page.getByTestId("chat-panel").waitFor({ timeout: 20_000 });
await pace(1500);

await ask(page, "which movie is Papanasam a remake of?");
await page.getByTestId("original-flag").first().waitFor({ timeout: 30_000 });
await pace(2500);
await page.getByTestId("version-card").first().scrollIntoViewIfNeeded();
await page
  .getByTestId("version-card")
  .first()
  .getByTestId("citations")
  .locator("summary")
  .click();
await pace(3000);

await ask(page, "Kaithi"); // the GS-02 decoy: honest no-match, zero cards
await page.getByTestId("turn").nth(1).waitFor({ timeout: 30_000 });
await page
  .getByTestId("turn")
  .nth(1)
  .getByTestId("assistant-answer")
  .waitFor({ timeout: 30_000 });
await pace(3000);

await ask(page, "no, the original one"); // GS-08 backtrack
await page
  .getByTestId("turn")
  .nth(2)
  .getByTestId("assistant-answer")
  .waitFor({ timeout: 30_000 });
await pace(2000);
const lastTrace = page.getByTestId("trace-view").last();
await lastTrace.scrollIntoViewIfNeeded();
await page.getByTestId("trace-view").first().locator("summary").click();
await pace(4000);

// --- Scene 3: STOP on camera ---
console.log("scene 3: firing gpu-stop (teardown on camera) …");
await run("uv", ["run", "python", "infra/gpu/jarvis.py", "nuke"], {
  cwd: REPO_ROOT,
  timeout: 240_000,
});
// The UI flips on its next 30 s status poll — the closing shot.
await page.getByTestId("offline-notice").waitFor({ timeout: 90_000 });
await page.getByTestId("status-pill").filter({ hasText: "offline by design" }).waitFor();
await pace(4000);
await page.getByTestId("attribution-footer").scrollIntoViewIfNeeded();
await pace(2500);

await context.close(); // flushes the video
const video = await page.video().path();
console.log("recorded:", video);
await browser.close();
