// Link builders (P6 task 5): one test per SourceId variant — the §4 unit list.
// Wikipedia links MUST carry the pinned revision id (?oldid=), the WP:Reusing-
// content attribution made revision-exact.
import { expect, test } from "vitest";
import { sourceLink } from "./citations";

test("wikidata → entity URL", () => {
  const link = sourceLink({ source: "wikidata", ref: "Q15401703" });
  expect(link.href).toBe("https://www.wikidata.org/wiki/Q15401703");
  expect(link.label).toBe("Wikidata Q15401703");
});

test("tmdb → movie URL with the tmdb: prefix stripped", () => {
  const link = sourceLink({ source: "tmdb", ref: "tmdb:266856" });
  expect(link.href).toBe("https://www.themoviedb.org/movie/266856");
});

test("tmdb bare id also builds", () => {
  expect(sourceLink({ source: "tmdb", ref: "266856" }).href).toBe(
    "https://www.themoviedb.org/movie/266856",
  );
});

test("imdb tt-id → title URL", () => {
  const link = sourceLink({ source: "imdb", ref: "tt3417422" });
  expect(link.href).toBe("https://www.imdb.com/title/tt3417422/");
});

test("wikipedia → en-wiki page URL pinned to the stored revision (oldid)", () => {
  const link = sourceLink({
    source: "wikipedia",
    ref: "Devadasu_(1953_film)@1348205009",
  });
  expect(link.href).toBe(
    "https://en.wikipedia.org/w/index.php?title=Devadasu_(1953_film)&oldid=1348205009",
  );
  expect(link.href).toContain("oldid=1348205009"); // the revision pin, explicitly
  expect(link.label).toBe("Wikipedia: Devadasu (1953 film)");
  expect(link.note).toContain("CC BY-SA 4.0");
});

test("wikipedia ref without a revision renders unlinked, never a guessed URL", () => {
  const link = sourceLink({ source: "wikipedia", ref: "Drishyam" });
  expect(link.href).toBeNull();
  expect(link.note).toContain("no revision recorded");
});

test("rule → no link, tooltip names the deterministic rule", () => {
  const link = sourceLink({
    source: "rule",
    ref: "dub-track-rule",
    field: "edge_type",
  });
  expect(link.href).toBeNull();
  expect(link.note).toBe(
    'derived by the documented rule "dub-track-rule" (field: edge_type)',
  );
});

test("human → no link, verification gate note", () => {
  const link = sourceLink({ source: "human", ref: "seed_slice@2026-07-02" });
  expect(link.href).toBeNull();
  expect(link.label).toBe("human-verified");
  expect(link.note).toContain("seed_slice@2026-07-02");
});

test("a source type outside the generated vocabulary is an explicit error", () => {
  const link = sourceLink({ source: "llm_guess", ref: "x" });
  expect(link.href).toBeNull();
  expect(link.label).toContain("unknown source type: llm_guess");
  expect(link.note).toContain("not in the v0 source vocabulary");
});
