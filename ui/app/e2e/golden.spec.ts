// The seven named golden regressions on the RENDERED DOM (P6_SPEC §4) — live
// mode: real app + real guardrails + seeded graph, scripted fake LLM driving
// the pinned expected tool flows. What the gating story promises, a browser sees.
import { expect, test, type Page } from "@playwright/test";

async function ask(page: Page, query: string) {
  await page.getByLabel("your message").fill(query);
  await page.getByRole("button", { name: "Send" }).click();
}

async function openChat(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("chat-panel")).toBeVisible();
}

test("test_ui_version_set_recall_gs01: all five Drishyam versions, original flagged", async ({
  page,
}) => {
  await openChat(page);
  await ask(page, "show me every version of Drishyam");
  const cards = page.getByTestId("version-card");
  await expect(cards).toHaveCount(5); // rendered version-set recall = 1.0
  // The original is flagged, first, and is the Malayalam 2013 film.
  const first = cards.first();
  await expect(first.getByTestId("original-flag")).toBeVisible();
  await expect(first).toContainText("Drishyam");
  await expect(first).toContainText("Malayalam · 2013");
  // Exactly one ORIGINAL flag across the whole set.
  await expect(page.getByTestId("original-flag")).toHaveCount(1);
  // All four remake languages present.
  for (const fact of ["Kannada · 2014", "Telugu · 2014", "Tamil · 2015", "Hindi · 2015"]) {
    await expect(page.getByTestId("version-card").filter({ hasText: fact })).toHaveCount(1);
  }
});

test("test_ui_version_set_recall_gs06: sequel traversal renders, badged sequel never remake", async ({
  page,
}) => {
  await openChat(page);
  await ask(page, "the whole Drishyam franchise, sequels included");
  // The sequel WORK surfaces via traversal: its Malayalam original is badged
  // sequel RELATIVE TO the queried work (v0 semantics), its own remakes as remakes.
  const sequelCard = page
    .getByTestId("version-card")
    .filter({ hasText: "Drishyam 2" })
    .filter({ hasText: "Malayalam" });
  await expect(sequelCard).toHaveCount(1);
  await expect(sequelCard.getByTestId("relationship-badge")).toHaveText("sequel");
  // The franchise renders at least the base five + the sequel lineage.
  expect(await page.getByTestId("version-card").count()).toBeGreaterThanOrEqual(6);
  // And nothing in the sequel lineage is mislabelled a remake OF THE ORIGINAL:
  // the Hindi Drishyam 2 is a remake of the sequel, rendered under the sequel set.
  const hindiSequel = page
    .getByTestId("version-card")
    .filter({ hasText: "Drishyam 2" })
    .filter({ hasText: "Hindi" });
  await expect(hindiSequel.getByTestId("relationship-badge")).toHaveText("remake");
});

test("test_ui_no_hallucinated_movie_gs02: decoy renders NO_MATCH, zero cards, no invented title", async ({
  page,
}) => {
  await openChat(page);
  await ask(page, "Kaithi");
  await expect(page.getByTestId("abstention")).toBeVisible();
  await expect(page.getByTestId("version-card")).toHaveCount(0);
  // Detector re-run over the RENDERED assistant text: no invented title claim —
  // the decoy title never appears in an answer, and no bold-asserted title at all.
  const answerText = await page.getByTestId("assistant-answer").innerText();
  expect(answerText).not.toContain("Kaithi");
  expect(answerText).not.toMatch(/\*\*[^*]+\*\*/);
  expect(answerText).not.toContain("2019");
});

test("test_ui_dub_vs_remake_gs04: Baahubali versions badge official dub; 'remake' on no card", async ({
  page,
}) => {
  await openChat(page);
  await ask(page, "all language versions of Baahubali: The Beginning");
  const cards = page.getByTestId("version-card");
  await expect(cards.first()).toBeVisible();
  const count = await cards.count();
  expect(count).toBeGreaterThanOrEqual(2); // the original + at least one dub
  for (let i = 0; i < count; i++) {
    const badge = cards.nth(i).getByTestId("relationship-badge");
    const text = (await badge.innerText()).toLowerCase();
    expect(["original", "official dub"]).toContain(text);
    expect(text).not.toContain("remake"); // the GS-04 wording rule, on the DOM
  }
});

test("test_ui_sibling_vs_remake_gs05: Devdas adaptations render as siblings, never a remake chain", async ({
  page,
}) => {
  await openChat(page);
  await ask(page, "the Devadasu movie");
  await expect(page.getByTestId("assistant-answer")).toContainText(
    "adaptations of a shared literary source",
  );
  // No card in the rendered set claims is_remake_of.
  const badges = page.getByTestId("relationship-badge");
  const count = await badges.count();
  for (let i = 0; i < count; i++) {
    expect((await badges.nth(i).innerText()).trim()).not.toBe("remake");
  }
});

test("test_ui_false_merge_gs10: Vikram renders two distinct works, never one merged set", async ({
  page,
}) => {
  await openChat(page);
  await ask(page, "Vikram Kamal Haasan");
  const answer = page.getByTestId("assistant-answer");
  // The disambiguation ask names BOTH distinct films — never merged.
  await expect(answer).toContainText("Vikram (1986)");
  await expect(answer).toContainText("Vikram (2022)");
  await expect(answer).toContainText("Which one");
  await expect(page.getByTestId("version-card")).toHaveCount(0); // no merged card set
  await expect(page.getByTestId("intent-chip")).toHaveText("disambiguate");
});

test("test_ui_backtracking_gs08: three turns over HTTP through the real session store", async ({
  page,
}) => {
  await openChat(page);

  // Turn 1: refine by actor → the Hindi Drishyam surfaces.
  await ask(page, "the Drishyam with Ajay Devgn");
  await expect(page.getByTestId("turn")).toHaveCount(1);
  let lastTurn = page.getByTestId("turn").last();
  await expect(lastTurn.getByTestId("version-card")).toHaveCount(1);
  await expect(lastTurn.getByTestId("version-card")).toContainText("Hindi · 2015");

  // Turn 2: the correction narrows to the ORIGINAL without losing the conversation.
  await ask(page, "no, the original one");
  await expect(page.getByTestId("turn")).toHaveCount(2);
  lastTurn = page.getByTestId("turn").last();
  await expect(lastTurn.getByTestId("version-card")).toHaveCount(1);
  await expect(lastTurn.getByTestId("version-card")).toContainText("Malayalam · 2013");
  await expect(lastTurn.getByTestId("original-flag")).toBeVisible();

  // Turn 3: still coherent — refine by language over the same standing set.
  await ask(page, "is there a Telugu one?");
  await expect(page.getByTestId("turn")).toHaveCount(3);
  lastTurn = page.getByTestId("turn").last();
  await expect(lastTurn.getByTestId("version-card")).toContainText("Drushyam");
  await expect(lastTurn.getByTestId("version-card")).toContainText("Telugu · 2014");

  // Turn 1's render is still on screen — the set was refined, not lost.
  await expect(page.getByTestId("turn").first().getByTestId("version-card")).toHaveCount(1);
});
