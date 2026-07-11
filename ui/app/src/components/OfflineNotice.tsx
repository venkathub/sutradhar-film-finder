// The structured GPU-off state (P6 task 3): "offline by design" is a first-class
// success screen — evidence references, the recorded demo video when present (key
// absent = no link, never a dead one), and a pointer to the replay browser.
import type { OfflineEvidence } from "../lib/api";

export default function OfflineNotice({
  detail,
  evidence,
}: {
  detail: string;
  evidence?: OfflineEvidence;
}) {
  return (
    <section className="offline-notice" data-testid="offline-notice">
      <h2>Live demo offline — by design</h2>
      <p className="offline-detail">{detail}</p>
      <p>
        The GPU is rented on-demand and brought up only for benchmarks and live
        demos; nothing inference-side runs 24/7. The story below replays from
        the recorded benchmark evidence instead.
      </p>
      {evidence && (
        <ul className="evidence-links">
          <li data-testid="benchmarks-ref">
            Benchmark report: <code>{evidence.benchmarks}</code> (in the
            repository)
          </li>
          {evidence.demo_video && (
            <li>
              <a href={evidence.demo_video} data-testid="demo-video-link">
                Watch the recorded demo video
              </a>
            </li>
          )}
        </ul>
      )}
    </section>
  );
}
