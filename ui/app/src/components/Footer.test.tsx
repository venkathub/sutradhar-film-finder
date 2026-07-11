// Attribution chrome (P6 task 5): the LICENSING.md obligations rendered and
// measured in a REAL browser — including the TMDB prominence condition (the
// TMDB logo must render less prominent than the Sutradhar mark).
import { expect, test } from "vitest";
import { render } from "vitest-browser-react";
import App from "../App";
import Footer, { IMDB_COURTESY, TMDB_NOTICE } from "./Footer";
import { OFF_STATUS, REPLAY_LIST, stubApi } from "../testing/stubs";

test("footer carries the exact TMDB notice + logo", async () => {
  const screen = await render(<Footer />);
  await expect
    .element(screen.getByTestId("tmdb-attribution"))
    .toHaveTextContent(TMDB_NOTICE);
  const logo = screen.getByTestId("tmdb-logo");
  await expect.element(logo).toBeVisible();
  await expect.element(logo).toHaveAttribute("alt", "TMDB");
  // The committed OFFICIAL asset (Vite inlines it as a data URL): the TMDB
  // brand gradient stops identify the real mark, not a hand-drawn substitute.
  const src = (await logo.element()).getAttribute("src") ?? "";
  expect(src).toContain("image/svg");
  for (const brandStop of ["90cea1", "3cbec9", "00b3e5"]) {
    expect(src).toContain(brandStop);
  }
});

test("footer carries the Wikipedia CC BY-SA 4.0 label with the license link", async () => {
  const screen = await render(<Footer />);
  const wiki = screen.getByTestId("wikipedia-attribution");
  await expect.element(wiki).toHaveTextContent("CC BY-SA 4.0");
  await expect
    .element(wiki.getByRole("link", { name: "CC BY-SA 4.0" }))
    .toHaveAttribute("href", "https://creativecommons.org/licenses/by-sa/4.0/");
});

test("footer carries the IMDb courtesy line + non-commercial note", async () => {
  const screen = await render(<Footer />);
  const imdb = screen.getByTestId("imdb-attribution");
  await expect.element(imdb).toHaveTextContent(IMDB_COURTESY);
  await expect.element(imdb).toHaveTextContent("non-commercial");
});

test("TMDB logo renders LESS PROMINENT than the Sutradhar mark (FAQ condition)", async () => {
  const api = stubApi({
    getStatus: () => Promise.resolve(OFF_STATUS),
    getReplays: () => Promise.resolve(REPLAY_LIST),
  });
  const screen = await render(<App api={api} />);
  const mark = await screen
    .getByRole("heading", { name: "Sutradhar" })
    .element();
  const logo = await screen.getByTestId("tmdb-logo").element();
  const markHeight = mark.getBoundingClientRect().height;
  const logoHeight = logo.getBoundingClientRect().height;
  expect(logoHeight).toBeGreaterThan(0); // actually rendered
  expect(logoHeight).toBeLessThan(markHeight); // measurably less prominent
});

test("attribution footer is present on every screen (off mode shown here)", async () => {
  const api = stubApi({
    getStatus: () => Promise.resolve(OFF_STATUS),
    getReplays: () => Promise.resolve(REPLAY_LIST),
  });
  const screen = await render(<App api={api} />);
  await expect.element(screen.getByTestId("attribution-footer")).toBeVisible();
});
