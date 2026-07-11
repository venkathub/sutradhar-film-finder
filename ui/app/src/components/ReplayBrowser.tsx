// Replay browser (P6 task 3): lists the pinned run's fixtures from GET
// /api/replays and renders a selected transcript through the SAME TurnList the
// live chat uses — the ROADMAP "when the GPU is off, the same story replays
// from recorded evidence" clause, as a component.
import { useEffect, useState } from "react";
import type { Api, Replay, ReplayList } from "../lib/api";
import { fromReplayTurn } from "../lib/turns";
import TurnList from "./TurnList";

export default function ReplayBrowser({ api }: { api: Api }) {
  const [list, setList] = useState<ReplayList | null>(null);
  const [replay, setReplay] = useState<Replay | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getReplays()
      .then((replays) => {
        if (!cancelled) setList(replays);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [api]);

  async function select(fixtureId: string) {
    try {
      setReplay(await api.getReplay(fixtureId));
      setError(null);
    } catch (err: unknown) {
      setError(String(err));
    }
  }

  if (error) {
    return (
      <section className="replay-browser">
        <p role="alert">Replay unavailable: {error}</p>
      </section>
    );
  }
  if (!list) {
    return (
      <section className="replay-browser">
        <p>Loading recorded replays…</p>
      </section>
    );
  }
  return (
    <section className="replay-browser" data-testid="replay-browser">
      <h2>Recorded replays</h2>
      <p className="run-stamp" data-testid="run-stamp">
        Pinned run <code>{list.run_id}</code> · model <code>{list.model}</code>{" "}
        · prompt <code>{list.prompt_hash.slice(0, 8)}</code>
      </p>
      <div className="replay-picker" role="group" aria-label="recorded fixtures">
        {list.available.map((fixtureId) => (
          <button
            key={fixtureId}
            type="button"
            className={
              replay?.fixture_id === fixtureId ? "fixture selected" : "fixture"
            }
            onClick={() => void select(fixtureId)}
          >
            {fixtureId}
          </button>
        ))}
      </div>
      {replay && (
        <div data-testid="replay-transcript">
          <TurnList turns={replay.turns.map(fromReplayTurn)} />
        </div>
      )}
    </section>
  );
}
