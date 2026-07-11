// The single turn render model (P6 task 3, §2.2): live ChatResponse turns and
// replayed pinned-run turns both map onto TurnView — ONE rendering path, one
// set of components, one set of tests.
import type {
  ChatUp,
  Citation,
  IntentPayload,
  ReplayTurn,
  TraceStep,
  Usage,
  VersionPayload,
} from "./api";

export interface TurnView {
  user: string;
  answer: string;
  intent: IntentPayload | null;
  versions: VersionPayload[];
  citations: Citation[];
  warnings: string[];
  trace: TraceStep[];
  latencyMs: number;
  /** Live turns carry tokens + cost (P6_SPEC §1.1); replayed turns: null —
   * per-turn usage was not recorded in the pinned transcripts, never faked. */
  usage: Usage | null;
  replayed: boolean;
}

/** The answer body without the machine-readable INTENT preamble line (the intent
 * is rendered separately as a chip; the preamble itself is envelope, not prose). */
export function displayAnswer(turn: TurnView): string {
  if (turn.intent !== null && turn.answer.startsWith("INTENT:")) {
    const newline = turn.answer.indexOf("\n");
    return newline === -1 ? "" : turn.answer.slice(newline + 1).trim();
  }
  return turn.answer;
}

/** Abstention (P6 task 4): the prompt contract marks a no-match answer with the
 * literal NO_MATCH token (system prompt v1.x, verified in the pinned GS-07a turn).
 * A NO_MATCH turn renders the abstention state and ZERO version cards. */
export function isNoMatch(turn: TurnView): boolean {
  return turn.answer.includes("NO_MATCH");
}

/** Low confidence (DEC-P2-5, made visual): a validated search_by_plot step reported
 * abstain=true (retrieval scored below the calibrated threshold) yet the answer still
 * names a match — render "low confidence", never fabricated certainty. */
export function isLowConfidence(turn: TurnView): boolean {
  if (isNoMatch(turn)) return false;
  return turn.trace.some(
    (step) =>
      step.valid &&
      step.tool === "search_by_plot" &&
      step.result_summary["abstain"] === true,
  );
}

export function fromChatUp(userMessage: string, response: ChatUp): TurnView {
  return {
    user: userMessage,
    answer: response.answer,
    intent: response.intent,
    versions: response.versions,
    citations: response.citations,
    warnings: response.warnings,
    trace: response.trace,
    latencyMs: response.latency_ms,
    usage: response.usage,
    replayed: false,
  };
}

export function fromReplayTurn(turn: ReplayTurn): TurnView {
  return {
    user: turn.message,
    answer: turn.answer,
    intent: turn.intent,
    versions: turn.versions,
    citations: turn.citations,
    warnings: turn.warnings,
    trace: turn.trace,
    latencyMs: turn.latency_ms,
    usage: null, // not recorded per turn in the pinned transcripts
    replayed: true,
  };
}
