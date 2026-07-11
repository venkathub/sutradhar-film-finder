// TraceView (P6 task 6): generated labels only, honest invalid-call states,
// bounded summaries, live-vs-replay latency honesty, usage/cost line.
import { expect, test } from "vitest";
import { render } from "vitest-browser-react";
import TraceView from "./TraceView";
import type { TraceStep } from "../lib/api";

function step(overrides: Partial<TraceStep>): TraceStep {
  return {
    step: 1,
    tool: "get_versions",
    arguments: { work_id: "w-1", scope: "indian" },
    valid: true,
    validation_error: null,
    result_summary: { kind: "versions", count: 5, ids: [] },
    latency_ms: 210.4,
    ...overrides,
  };
}

test("renders the GENERATED tool label, arguments, ✓ and the bounded summary", async () => {
  const screen = await render(
    <TraceView trace={[step({})]} usage={null} replayed={false} />,
  );
  await screen.getByText(/How this answer was assembled/).click();
  await expect
    .element(screen.getByTestId("trace-tool"))
    .toHaveTextContent("Get versions"); // the generated label, not the raw name
  await expect
    .element(screen.getByTestId("trace-args"))
    .toHaveTextContent('work_id="w-1", scope="indian"');
  await expect.element(screen.getByTestId("trace-valid")).toBeVisible();
  await expect
    .element(screen.getByTestId("trace-summary"))
    .toHaveTextContent("versions · 5");
  await expect.element(screen.getByText("210.4 ms")).toBeVisible();
});

test("unknown tool name renders an explicit error state, never a silent label", async () => {
  const screen = await render(
    <TraceView
      trace={[step({ tool: "delete_database", valid: false })]}
      usage={null}
      replayed={false}
    />,
  );
  await screen.getByText(/How this answer was assembled/).click();
  await expect
    .element(screen.getByTestId("trace-tool-unknown"))
    .toHaveTextContent("unknown tool: delete_database (not in tool_schema.v0)");
});

test("invalid call renders ✗ with the fed-back validation error", async () => {
  const screen = await render(
    <TraceView
      trace={[
        step({
          valid: false,
          validation_error: "hallucinated tool 'delete_database'",
          result_summary: { kind: "error", error: "hallucinated tool" },
        }),
      ]}
      usage={null}
      replayed={false}
    />,
  );
  await screen.getByText(/How this answer was assembled/).click();
  await expect
    .element(screen.getByTestId("trace-invalid"))
    .toHaveTextContent("rejected before execution: hallucinated tool 'delete_database'");
  await expect
    .element(screen.getByTestId("trace-summary"))
    .toHaveTextContent("error: hallucinated tool");
});

test("abstained retrieval is visible in the summary (DEC-P2-5 in the trace)", async () => {
  const screen = await render(
    <TraceView
      trace={[
        step({
          tool: "search_by_plot",
          arguments: { description: "father hides evidence" },
          result_summary: { kind: "results", count: 5, ids: [], abstain: true },
        }),
      ]}
      usage={null}
      replayed={false}
    />,
  );
  await screen.getByText(/How this answer was assembled/).click();
  await expect
    .element(screen.getByTestId("trace-summary"))
    .toHaveTextContent("results · 5 · abstained");
});

test("replayed steps suppress per-call latency (0.0 was never measured)", async () => {
  const screen = await render(
    <TraceView
      trace={[step({ latency_ms: 0.0 })]}
      usage={null}
      replayed={true}
    />,
  );
  await screen.getByText(/How this answer was assembled/).click();
  expect(screen.container.querySelector(".trace-latency")).toBeNull();
});

test("usage/cost line renders for live turns; view absent when nothing to show", async () => {
  const withUsage = await render(
    <TraceView
      trace={[step({})]}
      usage={{ prompt_tokens: 812, completion_tokens: 96, cost_usd: 0.000114 }}
      replayed={false}
    />,
  );
  await withUsage.getByText(/How this answer was assembled/).click();
  await expect
    .element(withUsage.getByTestId("trace-usage"))
    .toHaveTextContent("812 prompt + 96 completion tokens · $0.000114 (amortized GPU)");

  const empty = await render(
    <TraceView trace={[]} usage={null} replayed={true} />,
  );
  expect(empty.container.querySelector('[data-testid="trace-view"]')).toBeNull();
});

test("malformed-JSON arguments render honestly", async () => {
  const screen = await render(
    <TraceView
      trace={[step({ arguments: null, valid: false, validation_error: "arguments are not valid JSON" })]}
      usage={null}
      replayed={false}
    />,
  );
  await screen.getByText(/How this answer was assembled/).click();
  await expect
    .element(screen.getByText("(arguments did not parse as JSON)"))
    .toBeVisible();
});
