// App shell + mode switch (P6 tasks 2–3): the shell renders in a REAL browser
// (Vitest 4 Browser Mode, Playwright chromium); /api/status picks the mode —
// GPU off (the default) shows the offline notice + replay browser, up shows
// the chat panel. The tool-label map stays the generated v0 artifact.
import { expect, test } from "vitest";
import { render } from "vitest-browser-react";
import App from "./App";
import toolLabels from "./generated/tool_labels.json";
import { OFF_STATUS, REPLAY_LIST, UP_STATUS, stubApi } from "./testing/stubs";

const V0_TOOLS = [
  "get_versions",
  "get_work",
  "refine_filter",
  "resolve_title",
  "search_by_plot",
];

test("off mode (the default): mark, offline notice, replay browser", async () => {
  const api = stubApi({
    getStatus: () => Promise.resolve(OFF_STATUS),
    getReplays: () => Promise.resolve(REPLAY_LIST),
  });
  const screen = await render(<App api={api} />);
  await expect
    .element(screen.getByRole("heading", { name: "Sutradhar" }))
    .toBeVisible();
  await expect.element(screen.getByTestId("offline-notice")).toBeVisible();
  await expect.element(screen.getByTestId("replay-browser")).toBeVisible();
  await expect
    .element(screen.getByTestId("status-pill"))
    .toHaveTextContent("offline by design");
});

test("up mode: chat panel instead of the offline state", async () => {
  const api = stubApi({ getStatus: () => Promise.resolve(UP_STATUS) });
  const screen = await render(<App api={api} />);
  await expect.element(screen.getByTestId("chat-panel")).toBeVisible();
  await expect
    .element(screen.getByTestId("status-pill"))
    .toHaveTextContent("live (GPU window up)");
});

test("tool-label map carries exactly the five v0 tools", () => {
  expect(Object.keys(toolLabels.tools).sort()).toEqual(V0_TOOLS);
  expect(toolLabels.schema_version).toBe("v0");
  // Byte-derivation provenance: the generator stamps the artifact's sha256.
  expect(toolLabels.schema_sha256).toMatch(/^[0-9a-f]{64}$/);
});

test("labels are derived, never free text", () => {
  for (const [name, tool] of Object.entries(toolLabels.tools)) {
    // The deterministic rule: underscores → spaces, first letter capitalized.
    const derived = name.replaceAll("_", " ");
    expect(tool.label).toBe(derived.charAt(0).toUpperCase() + derived.slice(1));
    expect(tool.description.length).toBeGreaterThan(0);
  }
});
