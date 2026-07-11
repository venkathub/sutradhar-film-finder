// test_ui_gpu_off_replay (P6_SPEC §4): GPU off (the DEFAULT) → offline state →
// the replay browser plays the pinned GS-08a story, trace view included — the
// ROADMAP "when the GPU is off, the same story replays from recorded evidence".
import { expect, test } from "@playwright/test";

const OFF_BASE = "http://127.0.0.1:8766";

test("test_ui_gpu_off_replay: offline state, then GS-08a replays with trace view", async ({
  page,
}) => {
  await page.goto(OFF_BASE + "/");

  // The degradation state is a first-class screen, not an error.
  await expect(page.getByTestId("offline-notice")).toBeVisible();
  await expect(page.getByTestId("status-pill")).toHaveText("offline by design");
  await expect(page.getByTestId("benchmarks-ref")).toContainText("docs/BENCHMARKS.md");

  // The replay browser lists the pinned run's fixtures with the run stamp.
  await expect(page.getByTestId("run-stamp")).toContainText("20260704T093206Z-e9598564");
  await page.getByRole("button", { name: "GS-08a", exact: true }).click();

  // The full three-turn transcript renders through the SAME components as live.
  const transcript = page.getByTestId("replay-transcript");
  await expect(transcript.getByTestId("turn")).toHaveCount(3);
  await expect(transcript.getByTestId("user-message").first()).toHaveText(
    "the Drishyam with Ajay Devgn",
  );
  // Turn 2 rendered the full recorded version set with the original flagged.
  const turn2 = transcript.getByTestId("turn").nth(1);
  await expect(turn2.getByTestId("version-card")).toHaveCount(5);
  await expect(turn2.getByTestId("original-flag")).toBeVisible();
  // Recorded GPU latency shown honestly as replayed.
  await expect(transcript.getByText(/replayed · recorded GPU latency/).first()).toBeVisible();

  // The trace view renders the validated v0 calls with GENERATED labels.
  const trace = turn2.getByTestId("trace-view");
  await trace.locator("summary").click();
  await expect(trace.getByTestId("trace-tool")).toHaveText("Get versions");
  await expect(trace.getByTestId("trace-valid")).toBeVisible();

  // Attribution chrome present on the offline screen too.
  await expect(page.getByTestId("attribution-footer")).toBeVisible();
});
