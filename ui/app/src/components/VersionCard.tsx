// One language version as a card (P6 task 4 — the gating-story component).
// Relationship badges come from the GENERATED v0 vocabulary (tool_labels.json
// $defs.relationship): a null relationship renders honestly as "unverified
// relationship" (never a guessed label); a value OUTSIDE the artifact renders
// an explicit error badge, never a silent invention.
import type { VersionPayload } from "../lib/api";
import { sourceLink } from "../lib/citations";
import toolLabels from "../generated/tool_labels.json";

const RELATIONSHIPS: Record<string, { badge: string }> = toolLabels.relationships;

// Display chrome only (labels stay English — P6 non-goal): ISO 639-1 codes for
// the catalog's languages; unknown codes fall back to the code itself.
const LANGUAGE_NAMES: Record<string, string> = {
  ml: "Malayalam",
  ta: "Tamil",
  te: "Telugu",
  kn: "Kannada",
  hi: "Hindi",
  bn: "Bengali",
  mr: "Marathi",
  pa: "Punjabi",
  gu: "Gujarati",
  or: "Odia",
  en: "English",
};

export function languageName(code: string | null): string | null {
  if (code === null) return null;
  return LANGUAGE_NAMES[code] ?? code;
}

function RelationshipBadge({
  relationship,
}: {
  relationship: string | null;
}) {
  if (relationship === null) {
    return (
      <span className="badge badge-unverified" data-testid="relationship-badge">
        unverified relationship
      </span>
    );
  }
  const known = RELATIONSHIPS[relationship];
  if (!known) {
    return (
      <span className="badge badge-error" data-testid="relationship-badge">
        unknown edge: {relationship}
      </span>
    );
  }
  return (
    <span
      className={`badge badge-rel badge-${relationship}`}
      data-testid="relationship-badge"
    >
      {known.badge}
    </span>
  );
}

export default function VersionCard({ version }: { version: VersionPayload }) {
  return (
    <article
      className={version.is_original ? "version-card original" : "version-card"}
      data-testid="version-card"
      data-version-id={version.version_id}
    >
      <header className="version-card-head">
        <h3 className="version-title">{version.title}</h3>
        {version.is_original && (
          <span className="badge badge-original" data-testid="original-flag">
            ORIGINAL
          </span>
        )}
      </header>
      <p className="version-facts">
        {[languageName(version.language), version.year]
          .filter((fact) => fact !== null)
          .join(" · ")}
      </p>
      <RelationshipBadge relationship={version.relationship} />
      {version.confidence && (
        <span
          className={`badge badge-confidence badge-${version.confidence.toLowerCase()}`}
          data-testid="confidence-badge"
        >
          {version.confidence}
        </span>
      )}
      {version.cast_lead.length > 0 && (
        <p className="version-cast" data-testid="version-cast">
          {version.cast_lead.join(", ")}
        </p>
      )}
      {version.sources.length > 0 && (
        <details className="citations" data-testid="citations">
          <summary>
            {version.sources.length}{" "}
            {version.sources.length === 1 ? "source" : "sources"}
          </summary>
          <ul className="citation-list">
            {version.sources.map((source, i) => {
              const link = sourceLink(source);
              return (
                <li key={i} data-testid="citation">
                  {link.href ? (
                    <a href={link.href} target="_blank" rel="noreferrer">
                      {link.label}
                    </a>
                  ) : (
                    <span className="citation-unlinked">{link.label}</span>
                  )}
                  {link.note && (
                    <span className="citation-note" title={link.note}>
                      {" "}
                      — {link.note}
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        </details>
      )}
    </article>
  );
}
