// One turn list for BOTH live and replayed turns (P6 task 3, §2.2).
// Task-3 scope: user bubble, intent chip, answer text, warnings, latency.
// Version cards (task 4), citations (task 5), and the trace view (task 6)
// plug into this same component — TurnView already carries their data.
import type { TurnView } from "../lib/turns";
import { displayAnswer } from "../lib/turns";
import TraceView from "./TraceView";
import VersionSet from "./VersionSet";

export default function TurnList({ turns }: { turns: TurnView[] }) {
  return (
    <ol className="turn-list" aria-label="conversation">
      {turns.map((turn, i) => (
        <li className="turn" key={i} data-testid="turn">
          <div className="bubble user" data-testid="user-message">
            {turn.user}
          </div>
          <div
            className="bubble assistant"
            data-testid="assistant-answer"
            data-replayed={turn.replayed || undefined}
          >
            {turn.intent && (
              <span className="intent-chip" data-testid="intent-chip">
                {turn.intent.intent}
              </span>
            )}
            <p className="answer-text">{displayAnswer(turn)}</p>
            <VersionSet turn={turn} />
            {turn.warnings.length > 0 && (
              <ul className="warnings" data-testid="warnings">
                {turn.warnings.map((warning, w) => (
                  <li key={w}>{warning}</li>
                ))}
              </ul>
            )}
            <TraceView
              trace={turn.trace}
              usage={turn.usage}
              replayed={turn.replayed}
            />
            <span className="turn-meta">
              {turn.replayed ? "replayed · recorded GPU latency " : ""}
              {turn.latencyMs > 0 ? `${Math.round(turn.latencyMs)} ms` : ""}
            </span>
          </div>
        </li>
      ))}
    </ol>
  );
}
