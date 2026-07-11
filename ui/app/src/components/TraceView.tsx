// Trace view (P6 task 6, DEC-P6-4): how the answer was assembled — rendered
// from what the ORCHESTRATOR already validated, never re-derived. Tool and
// param labels come from the GENERATED v0 map (tool_labels.json): a tool name
// outside it renders an explicit error state, not a silent label. Live and
// replayed turns share this one component (TurnView.trace).
// Langfuse trace ids stay server-side (D4): the ops view never leaks into the
// browser, so no observability credentials or links render here.
import type { TraceStep, Usage } from "../lib/api";
import toolLabels from "../generated/tool_labels.json";

type ToolInfo = {
  label: string;
  description: string;
  params: Record<string, { label: string; required: boolean }>;
};

const TOOLS: Record<string, ToolInfo> = toolLabels.tools;

function Arguments({ step }: { step: TraceStep }) {
  if (step.arguments === null) {
    return (
      <span className="trace-args trace-args-unparsed">
        (arguments did not parse as JSON)
      </span>
    );
  }
  const entries = Object.entries(step.arguments);
  if (entries.length === 0) return null;
  const params = TOOLS[step.tool]?.params ?? {};
  return (
    <span className="trace-args" data-testid="trace-args">
      {entries.map(([key, value], i) => (
        <span key={key} className="trace-arg">
          {i > 0 && ", "}
          <span title={params[key]?.label ?? key}>{key}</span>=
          {JSON.stringify(value)}
        </span>
      ))}
    </span>
  );
}

function Summary({ step }: { step: TraceStep }) {
  const summary = step.result_summary;
  const kind = String(summary["kind"] ?? "");
  if (kind === "error") {
    return (
      <span className="trace-summary trace-error" data-testid="trace-summary">
        error: {String(summary["error"] ?? "")}
      </span>
    );
  }
  const count = summary["count"];
  return (
    <span className="trace-summary" data-testid="trace-summary">
      {kind}
      {typeof count === "number" ? ` · ${count}` : ""}
      {summary["abstain"] === true ? " · abstained" : ""}
    </span>
  );
}

function StepRow({ step, replayed }: { step: TraceStep; replayed: boolean }) {
  const tool = TOOLS[step.tool];
  return (
    <li className="trace-step" data-testid="trace-step">
      <span className="trace-step-no">{step.step}.</span>{" "}
      {tool ? (
        <span
          className="trace-tool"
          data-testid="trace-tool"
          title={tool.description}
        >
          {tool.label}
        </span>
      ) : (
        <span className="trace-tool trace-error" data-testid="trace-tool-unknown">
          unknown tool: {step.tool} (not in tool_schema.v0)
        </span>
      )}{" "}
      <Arguments step={step} />
      {step.valid ? (
        <span className="trace-valid" data-testid="trace-valid" title="validated against tool_schema.v0.json">
          ✓
        </span>
      ) : (
        <span className="trace-invalid" data-testid="trace-invalid">
          ✗ rejected before execution
          {step.validation_error ? `: ${step.validation_error}` : ""}
        </span>
      )}{" "}
      <Summary step={step} />
      {!replayed && (
        <span className="trace-latency">{step.latency_ms.toFixed(1)} ms</span>
      )}
    </li>
  );
}

export default function TraceView({
  trace,
  usage,
  replayed,
}: {
  trace: TraceStep[];
  usage: Usage | null;
  replayed: boolean;
}) {
  if (trace.length === 0 && usage === null) return null;
  return (
    <details className="trace-view" data-testid="trace-view">
      <summary>
        How this answer was assembled — {trace.length}{" "}
        {trace.length === 1 ? "tool call" : "tool calls"}
      </summary>
      {trace.length > 0 && (
        <ol className="trace-steps">
          {trace.map((step) => (
            <StepRow key={step.step} step={step} replayed={replayed} />
          ))}
        </ol>
      )}
      {usage && (
        <p className="trace-usage" data-testid="trace-usage">
          {usage.prompt_tokens} prompt + {usage.completion_tokens} completion
          tokens
          {usage.cost_usd !== null
            ? ` · $${usage.cost_usd.toFixed(6)} (amortized GPU)`
            : ""}
        </p>
      )}
    </details>
  );
}
