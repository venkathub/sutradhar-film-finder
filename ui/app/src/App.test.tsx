// Scaffold smoke (P6 task 2): the shell renders in a REAL browser (Vitest 4
// Browser Mode, Playwright chromium) and the tool-label map is the generated
// v0 artifact — exactly the five tools, nothing hand-added.
import { expect, test } from "vitest";
import { render } from "vitest-browser-react";
import App from "./App";
import toolLabels from "./generated/tool_labels.json";

const V0_TOOLS = [
  "get_versions",
  "get_work",
  "refine_filter",
  "resolve_title",
  "search_by_plot",
];

test("renders the Sutradhar mark and shell", async () => {
  const screen = await render(<App />);
  await expect
    .element(screen.getByRole("heading", { name: "Sutradhar" }))
    .toBeVisible();
  await expect
    .element(screen.getByTestId("scaffold-placeholder"))
    .toBeVisible();
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
