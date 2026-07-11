// App shell (P6 tasks 2–3). Picks live vs offline mode from /api/status
// (polled at the server's own 30 s cache TTL, DEC-P5-5) — GPU off is the
// DEFAULT and a first-class screen: offline notice + replay browser. The
// trace-view labels come from the generated v0 artifact, never typed by hand.
import { useEffect, useState } from "react";
import OfflineNotice from "./components/OfflineNotice";
import ChatPanel from "./components/ChatPanel";
import ReplayBrowser from "./components/ReplayBrowser";
import type { Api, StatusResponse } from "./lib/api";
import { httpApi } from "./lib/api";

const STATUS_POLL_MS = 30_000; // matches the server-side StatusCache TTL

export default function App({ api = httpApi }: { api?: Api }) {
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    const refresh = () =>
      api
        .getStatus()
        .then((next) => {
          if (!cancelled) setStatus(next);
        })
        .catch(() => {
          if (!cancelled)
            setStatus({ status: "off", detail: "API unreachable" });
        });
    void refresh();
    const timer = setInterval(() => void refresh(), STATUS_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [api]);

  return (
    <div className="app">
      <header className="app-header">
        {/* The Sutradhar mark — the app's PRIMARY mark (the TMDB logo added in
            task 5 must render less prominent than this, per the TMDB FAQ). */}
        <h1 className="mark">Sutradhar</h1>
        <p className="tagline">
          Find an Indian film from its story, plot, or cast — every language
          version, the original flagged, every claim cited.
        </p>
        {status && (
          <span
            className={`status-pill status-${status.status}`}
            data-testid="status-pill"
          >
            {status.status === "up" ? "live (GPU window up)" : "offline by design"}
          </span>
        )}
      </header>
      <main className="app-main">
        {!status && <p className="placeholder">Checking API status…</p>}
        {status?.status === "up" && <ChatPanel api={api} />}
        {status && status.status !== "up" && (
          <>
            <OfflineNotice detail={status.detail} evidence={status.evidence} />
            <ReplayBrowser api={api} />
          </>
        )}
      </main>
    </div>
  );
}
