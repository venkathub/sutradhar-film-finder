// Per-claim citation link builders (P6 task 5, P6_SPEC §2.2). Pure functions,
// one per SourceId variant — the vocabulary itself comes from the GENERATED
// tool_labels.json (source_types, from $defs.sources in the v0 artifact); a
// source type outside it renders an explicit error entry, never a silent link.
import type { SourceRef } from "./api";
import toolLabels from "../generated/tool_labels.json";

const SOURCE_TYPES: readonly string[] = toolLabels.source_types;

export interface CitationLink {
  /** null = deliberately unlinked (rule / human / unknown) — rendered as a note. */
  href: string | null;
  label: string;
  note: string | null;
}

/** Wikipedia refs are `<page-title>@<revision>` (SourceRef contract; verified in the
 * live graph, e.g. `Devadasu_(1953_film)@1348205009`). The ref carries no wiki
 * language: en-wiki was the P1 extraction wiki, so links target en.wikipedia.org —
 * revision-pinned via ?oldid= (the LICENSING.md CC BY-SA attribution obligation). */
function wikipediaLink(ref: string): CitationLink {
  const at = ref.lastIndexOf("@");
  if (at <= 0 || at === ref.length - 1) {
    return {
      href: null,
      label: `wikipedia: ${ref}`,
      note: "unpinned reference (no revision recorded)",
    };
  }
  const title = ref.slice(0, at);
  const revision = ref.slice(at + 1);
  return {
    href: `https://en.wikipedia.org/w/index.php?title=${encodeURIComponent(title)}&oldid=${encodeURIComponent(revision)}`,
    label: `Wikipedia: ${title.replaceAll("_", " ")}`,
    note: `revision ${revision} (CC BY-SA 4.0)`,
  };
}

export function sourceLink(source: SourceRef): CitationLink {
  const ref = typeof source.ref === "string" ? source.ref : "";
  switch (source.source) {
    case "wikidata":
      return {
        href: `https://www.wikidata.org/wiki/${encodeURIComponent(ref)}`,
        label: `Wikidata ${ref}`,
        note: null,
      };
    case "tmdb": {
      const id = ref.startsWith("tmdb:") ? ref.slice("tmdb:".length) : ref;
      return {
        href: `https://www.themoviedb.org/movie/${encodeURIComponent(id)}`,
        label: `TMDB ${id}`,
        note: null,
      };
    }
    case "imdb":
      return {
        href: `https://www.imdb.com/title/${encodeURIComponent(ref)}/`,
        label: `IMDb ${ref}`,
        note: null,
      };
    case "wikipedia":
      return wikipediaLink(ref);
    case "rule":
      return {
        href: null,
        label: "deterministic rule",
        note:
          `derived by the documented rule "${ref}"` +
          (source.field ? ` (field: ${String(source.field)})` : ""),
      };
    case "human":
      return {
        href: null,
        label: "human-verified",
        note: `passed the human verification gate (${ref})`,
      };
    default:
      // Not in the generated $defs.sources vocabulary — explicit error state.
      return {
        href: null,
        label: `unknown source type: ${String(source.source)}`,
        note: SOURCE_TYPES.includes(String(source.source))
          ? "builder missing for a known source type" // unreachable by construction
          : "not in the v0 source vocabulary",
      };
  }
}
