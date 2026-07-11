// The single turn render model (P6 task 3, §2.2): live ChatResponse turns and
// replayed pinned-run turns both map onto TurnView — ONE rendering path, one
// set of components, one set of tests.
import type {
  ChatUp,
  Citation,
  IntentPayload,
  ReplayTurn,
  TraceStep,
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
    replayed: true,
  };
}
