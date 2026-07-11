// VersionCard (P6 task 4): the gating-story card — original flag, the five
// artifact-derived edge badges, the null → "unverified relationship" honesty
// rule, unknown-edge error state, confidence tiers.
import { expect, test } from "vitest";
import { render } from "vitest-browser-react";
import VersionCard from "./VersionCard";
import toolLabels from "../generated/tool_labels.json";
import { version } from "../testing/stubs";

test("original version carries the ORIGINAL flag", async () => {
  const screen = await render(<VersionCard version={version({})} />);
  await expect.element(screen.getByTestId("original-flag")).toBeVisible();
  await expect
    .element(screen.getByTestId("relationship-badge"))
    .toHaveTextContent("original");
});

test("non-original versions never show the flag", async () => {
  const screen = await render(
    <VersionCard
      version={version({ is_original: false, relationship: "is_remake_of" })}
    />,
  );
  expect(screen.container.querySelector('[data-testid="original-flag"]')).toBeNull();
});

// Data-driven over the GENERATED vocabulary: all five v0 edges render their badge.
for (const [relationship, { badge }] of Object.entries(toolLabels.relationships)) {
  test(`edge ${relationship} renders badge "${badge}"`, async () => {
    const screen = await render(
      <VersionCard version={version({ relationship, is_original: false })} />,
    );
    await expect
      .element(screen.getByTestId("relationship-badge"))
      .toHaveTextContent(badge);
  });
}

test("official dub badge never contains the word 'remake' (GS-04 wording)", async () => {
  const screen = await render(
    <VersionCard
      version={version({ relationship: "is_official_dub_of", is_original: false })}
    />,
  );
  const badge = screen.getByTestId("relationship-badge");
  await expect.element(badge).toHaveTextContent("official dub");
  expect((await badge.element()).textContent).not.toContain("remake");
});

test("relationship null renders 'unverified relationship', never a guess", async () => {
  const screen = await render(
    <VersionCard version={version({ relationship: null, is_original: false })} />,
  );
  await expect
    .element(screen.getByTestId("relationship-badge"))
    .toHaveTextContent("unverified relationship");
});

test("a value outside the v0 vocabulary is an explicit error, not a silent render", async () => {
  const screen = await render(
    <VersionCard
      version={version({ relationship: "is_inspired_by", is_original: false })}
    />,
  );
  await expect
    .element(screen.getByTestId("relationship-badge"))
    .toHaveTextContent("unknown edge: is_inspired_by");
});

test("confidence tiers render as badges (HIGH / MEDIUM)", async () => {
  const high = await render(<VersionCard version={version({})} />);
  await expect
    .element(high.getByTestId("confidence-badge"))
    .toHaveTextContent("HIGH");
  const medium = await render(
    <VersionCard version={version({ version_id: "v-m", confidence: "MEDIUM" })} />,
  );
  await expect
    .element(medium.getByTestId("confidence-badge").last())
    .toHaveTextContent("MEDIUM");
});

test("facts line: language display name + year; cast rendered", async () => {
  const screen = await render(<VersionCard version={version({})} />);
  await expect
    .element(screen.getByText("Malayalam · 2013"))
    .toBeVisible();
  await expect
    .element(screen.getByTestId("version-cast"))
    .toHaveTextContent("Mohanlal");
});

test("per-claim citations: wikidata source renders a clickable entity link", async () => {
  const screen = await render(<VersionCard version={version({})} />);
  const citations = screen.getByTestId("citations");
  await expect.element(citations).toHaveTextContent("1 source");
  await citations.getByText("1 source").click(); // open the disclosure
  await expect
    .element(citations.getByRole("link", { name: "Wikidata Q15401703" }))
    .toHaveAttribute("href", "https://www.wikidata.org/wiki/Q15401703");
});

test("rule source renders unlinked with the named-rule note", async () => {
  const screen = await render(
    <VersionCard
      version={version({
        sources: [{ source: "rule", ref: "dub-track-rule", field: "edge_type" }],
      })}
    />,
  );
  const citations = screen.getByTestId("citations");
  await citations.getByText("1 source").click();
  await expect
    .element(citations.getByTestId("citation"))
    .toHaveTextContent('documented rule "dub-track-rule"');
  expect(
    (await citations.getByTestId("citation").element()).querySelector("a"),
  ).toBeNull(); // deliberately unlinked
});
