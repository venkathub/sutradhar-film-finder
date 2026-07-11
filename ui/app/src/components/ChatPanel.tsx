// Live chat panel (P6 task 3). conversation_id from the first response is
// carried on every following request — the GS-08 backtracking mechanics over
// the DEC-P5-2 session store. Progress is DETERMINISTIC staged states (D2 /
// DEC-P6-2): no streaming; the ~5 s turn is covered by an honest indicator
// driven by a timer, not by fake server events.
import { useEffect, useRef, useState } from "react";
import type { Api, ChatError, ChatOff } from "../lib/api";
import type { TurnView } from "../lib/turns";
import { fromChatUp } from "../lib/turns";
import OfflineNotice from "./OfflineNotice";
import TurnList from "./TurnList";

const STAGES = ["parsing", "searching the graph", "composing"] as const;
const STAGE_MS = 1600;

export default function ChatPanel({
  api,
  onWentOffline,
}: {
  api: Api;
  onWentOffline?: () => void;
}) {
  const [turns, setTurns] = useState<TurnView[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [stage, setStage] = useState(0);
  const [aborted, setAborted] = useState<ChatOff | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);
  const conversationId = useRef<string | null>(null);

  useEffect(() => {
    if (!pending) return;
    setStage(0);
    const timer = setInterval(
      () => setStage((current) => Math.min(current + 1, STAGES.length - 1)),
      STAGE_MS,
    );
    return () => clearInterval(timer);
  }, [pending]);

  async function send() {
    const message = draft.trim();
    if (!message || pending) return;
    setDraft("");
    setPending(true);
    setRequestError(null);
    try {
      const result = await api.postChat({
        conversation_id: conversationId.current,
        message,
      });
      if ("error" in result) {
        setRequestError((result as ChatError).detail);
      } else if (result.status === "up") {
        conversationId.current = result.conversation_id;
        setTurns((current) => [...current, fromChatUp(message, result)]);
      } else {
        setAborted(result); // turn aborted mid-conversation → offline state
        onWentOffline?.();
      }
    } catch (err: unknown) {
      setRequestError(String(err));
    } finally {
      setPending(false);
    }
  }

  return (
    <section className="chat-panel" data-testid="chat-panel">
      <TurnList turns={turns} />
      {pending && (
        <p className="progress" role="status" data-testid="progress">
          {STAGES[stage]}…
        </p>
      )}
      {requestError && (
        <p className="request-error" role="alert">
          {requestError}
        </p>
      )}
      {aborted && (
        <OfflineNotice detail={aborted.detail} evidence={aborted.evidence} />
      )}
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          void send();
        }}
      >
        <input
          type="text"
          aria-label="your message"
          placeholder="Describe the film — any language, any script…"
          value={draft}
          disabled={pending}
          onChange={(event) => setDraft(event.target.value)}
        />
        <button type="submit" disabled={pending || draft.trim() === ""}>
          Send
        </button>
      </form>
    </section>
  );
}
