// VersionSet (P6 task 4): the honesty states — NO_MATCH → abstention with ZERO
// cards; abstain-with-results → "low confidence" banner (DEC-P2-5, visual);
// the GS-01 gate set renders all five versions with the original first.
import { expect, test } from "vitest";
import { render } from "vitest-browser-react";
import VersionSet from "./VersionSet";
import type { TraceStep } from "../lib/api";
import { DRISHYAM_SET, turnView } from "../testing/stubs";

function searchStep(abstain: boolean): TraceStep {
  return {
    step: 1,
    tool: "search_by_plot",
    arguments: { description: "father hides evidence" },
    valid: true,
    validation_error: null,
    result_summary: { kind: "results", count: 5, ids: [], abstain },
    latency_ms: 210.0,
  };
}

test("GS-01 shape: all five versions rendered, original flagged and first", async () => {
  const turn = turnView({ versions: DRISHYAM_SET });
  const screen = await render(<VersionSet turn={turn} />);
  const cards = screen.getByTestId("version-card");
  expect(await cards.all()).toHaveLength(5); // rendered version-set recall = 1.0
  await expect
    .element(cards.first().getByTestId("original-flag"))
    .toBeVisible();
  await expect
    .element(cards.first().getByText("Drishyam", { exact: true }))
    .toBeVisible();
});

test("NO_MATCH renders the abstention state and zero version cards", async () => {
  const turn = turnView({
    answer:
      'INTENT: {"intent": "out_of_catalog", "slots": {}}\n' +
      "I couldn't find a film matching that description. NO_MATCH.",
    intent: { intent: "out_of_catalog", slots: {} },
    versions: DRISHYAM_SET, // even if data slipped through, cards are suppressed
  });
  const screen = await render(<VersionSet turn={turn} />);
  await expect.element(screen.getByTestId("abstention")).toBeVisible();
  expect(screen.container.querySelector('[data-testid="version-card"]')).toBeNull();
});

test("abstain=true with results renders 'low confidence', never certainty", async () => {
  const turn = turnView({
    versions: DRISHYAM_SET,
    trace: [searchStep(true)],
  });
  const screen = await render(<VersionSet turn={turn} />);
  await expect.element(screen.getByTestId("low-confidence")).toBeVisible();
  expect(await screen.getByTestId("version-card").all()).toHaveLength(5);
});

test("abstain=false renders no low-confidence banner", async () => {
  const turn = turnView({ versions: DRISHYAM_SET, trace: [searchStep(false)] });
  const screen = await render(<VersionSet turn={turn} />);
  expect(
    screen.container.querySelector('[data-testid="low-confidence"]'),
  ).toBeNull();
});

test("no versions and no NO_MATCH renders nothing (answer-only turn)", async () => {
  const screen = await render(<VersionSet turn={turnView({})} />);
  expect(screen.container.querySelector('[data-testid="version-set"]')).toBeNull();
  expect(screen.container.querySelector('[data-testid="abstention"]')).toBeNull();
});
