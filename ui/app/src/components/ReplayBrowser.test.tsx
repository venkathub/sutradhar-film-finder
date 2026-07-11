// ReplayBrowser (P6 task 3): fixtures listed from GET /api/replays with the
// pinned-run stamp; a selected replay renders through the SAME TurnList the
// live chat uses (one rendering path).
import { expect, test } from "vitest";
import { render } from "vitest-browser-react";
import ReplayBrowser from "./ReplayBrowser";
import { GS08A_REPLAY, REPLAY_LIST, stubApi } from "../testing/stubs";

test("lists the pinned run's fixtures with the run stamp", async () => {
  const api = stubApi({ getReplays: () => Promise.resolve(REPLAY_LIST) });
  const screen = await render(<ReplayBrowser api={api} />);
  await expect
    .element(screen.getByTestId("run-stamp"))
    .toHaveTextContent("20260704T093206Z-e9598564");
  await expect
    .element(screen.getByRole("button", { name: "GS-08a" }))
    .toBeVisible();
  await expect
    .element(screen.getByRole("button", { name: "GS-01" }))
    .toBeVisible();
});

test("selecting a fixture renders its turns (user + answer, replay-marked)", async () => {
  const api = stubApi({
    getReplays: () => Promise.resolve(REPLAY_LIST),
    getReplay: (id) => {
      expect(id).toBe("GS-08a");
      return Promise.resolve(GS08A_REPLAY);
    },
  });
  const screen = await render(<ReplayBrowser api={api} />);
  await screen.getByRole("button", { name: "GS-08a" }).click();

  const transcript = screen.getByTestId("replay-transcript");
  await expect
    .element(transcript.getByTestId("user-message").first())
    .toHaveTextContent("the Drishyam with Ajay Devgn");
  // INTENT preamble stripped for display; intent shown as a chip instead.
  await expect
    .element(transcript.getByTestId("assistant-answer").last())
    .toHaveTextContent("The original is Drishyam (2013, Malayalam).");
  await expect
    .element(transcript.getByTestId("intent-chip").first())
    .toHaveTextContent("disambiguate");
  // Replayed turns are marked and carry the RECORDED GPU latency.
  await expect
    .element(transcript.getByTestId("assistant-answer").first())
    .toHaveAttribute("data-replayed", "true");
  await expect
    .element(transcript.getByText(/recorded GPU latency 1702 ms/))
    .toBeVisible();
});

test("replay API failure shows an error state, never a blank screen", async () => {
  const api = stubApi({
    getReplays: () => Promise.reject(new Error("boom")),
  });
  const screen = await render(<ReplayBrowser api={api} />);
  await expect.element(screen.getByRole("alert")).toHaveTextContent("boom");
});
