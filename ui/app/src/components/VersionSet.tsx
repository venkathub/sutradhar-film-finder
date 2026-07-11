// The version set under an answer (P6 task 4). Honesty states first:
// - NO_MATCH answer → abstention callout, ZERO cards (suppressed even if data
//   slipped through — a NO_MATCH with cards would contradict itself);
// - abstain=true with a named match (DEC-P2-5) → "low confidence" banner over
//   the cards, never fabricated certainty;
// - otherwise: cards in server order (original first — the orchestrator/adapter
//   already orders them; the UI never re-sorts or re-derives).
import type { TurnView } from "../lib/turns";
import { isLowConfidence, isNoMatch } from "../lib/turns";
import VersionCard from "./VersionCard";

export default function VersionSet({ turn }: { turn: TurnView }) {
  if (isNoMatch(turn)) {
    return (
      <p className="abstention" data-testid="abstention">
        No confident match in the catalog — Sutradhar abstains rather than
        guessing.
      </p>
    );
  }
  if (turn.versions.length === 0) return null;
  return (
    <div className="version-set" data-testid="version-set">
      {isLowConfidence(turn) && (
        <p className="low-confidence" data-testid="low-confidence">
          Low-confidence match — retrieval scored below the calibrated
          threshold; treat these as suggestions, not certainty.
        </p>
      )}
      <div className="version-cards">
        {turn.versions.map((version) => (
          <VersionCard key={version.version_id} version={version} />
        ))}
      </div>
    </div>
  );
}
