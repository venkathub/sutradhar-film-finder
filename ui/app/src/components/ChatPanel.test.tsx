// ChatPanel (P6 task 3): deterministic progress states while pending (D2),
// conversation_id carried across turns (the GS-08 mechanics), and a mid-
// conversation TurnAborted rendering the offline state — never a crash.
import { expect, test } from "vitest";
import { render } from "vitest-browser-react";
import ChatPanel from "./ChatPanel";
import type { ChatOff, ChatResult } from "../lib/api";
import { chatUp, stubApi } from "../testing/stubs";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

test("progress indicator while pending; answer rendered after resolve", async () => {
  const turn = deferred<ChatResult>();
  const api = stubApi({ postChat: () => turn.promise });
  const screen = await render(<ChatPanel api={api} />);

  await screen.getByLabelText("your message").fill("papanasam?");
  await screen.getByRole("button", { name: "Send" }).click();
  await expect.element(screen.getByTestId("progress")).toBeVisible();

  turn.resolve(chatUp({ answer: "Papanasam is a remake of Drishyam." }));
  await expect
    .element(screen.getByTestId("assistant-answer"))
    .toHaveTextContent("Papanasam is a remake of Drishyam.");
  expect(screen.container.querySelector('[data-testid="progress"]')).toBeNull();
});

test("turn 2 sends the conversation_id from turn 1 (GS-08 mechanics)", async () => {
  const bodies: Array<{ conversation_id: string | null; message: string }> = [];
  const api = stubApi({
    postChat: (body) => {
      bodies.push(body);
      return Promise.resolve(chatUp({ conversation_id: "conv-42" }));
    },
  });
  const screen = await render(<ChatPanel api={api} />);

  await screen.getByLabelText("your message").fill("which movie is papanasam a remake of?");
  await screen.getByRole("button", { name: "Send" }).click();
  await expect.element(screen.getByTestId("assistant-answer").first()).toBeVisible();

  await screen.getByLabelText("your message").fill("no, the original one");
  await screen.getByRole("button", { name: "Send" }).click();
  await expect.element(screen.getByTestId("assistant-answer").last()).toBeVisible();

  expect(bodies[0].conversation_id).toBeNull(); // new conversation
  expect(bodies[1].conversation_id).toBe("conv-42"); // carried across turns
  expect(bodies[1].message).toBe("no, the original one");
});

test("turn aborted mid-conversation renders the offline state", async () => {
  const off: ChatOff = {
    conversation_id: null,
    status: "off",
    detail:
      "Live demo offline by design — the GPU is on-demand. (turn aborted: died)",
    evidence: { benchmarks: "docs/BENCHMARKS.md", replay: "/api/replay/GS-08a" },
    request_live_demo: "see docs/RUNBOOK.md",
  };
  const api = stubApi({ postChat: () => Promise.resolve(off) });
  const screen = await render(<ChatPanel api={api} />);

  await screen.getByLabelText("your message").fill("papanasam?");
  await screen.getByRole("button", { name: "Send" }).click();
  await expect.element(screen.getByTestId("offline-notice")).toBeVisible();
  await expect
    .element(screen.getByTestId("offline-notice"))
    .toHaveTextContent("turn aborted");
});

test("structured 4xx (limits) renders as an alert, not a crash", async () => {
  const api = stubApi({
    postChat: () =>
      Promise.resolve({ error: "limit", detail: "conversation turn cap (20) reached" }),
  });
  const screen = await render(<ChatPanel api={api} />);
  await screen.getByLabelText("your message").fill("one more");
  await screen.getByRole("button", { name: "Send" }).click();
  await expect
    .element(screen.getByRole("alert"))
    .toHaveTextContent("turn cap (20) reached");
});
