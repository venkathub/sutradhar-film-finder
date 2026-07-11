// OfflineNotice (P6 task 3): the GPU-off screen renders the structured
// payload; the demo-video link appears ONLY when the evidence carries it
// (key absent = no dead link).
import { expect, test } from "vitest";
import { render } from "vitest-browser-react";
import OfflineNotice from "./OfflineNotice";
import { OFF_STATUS } from "../testing/stubs";

test("renders detail + benchmark reference; no video link when absent", async () => {
  const screen = await render(
    <OfflineNotice detail={OFF_STATUS.detail} evidence={OFF_STATUS.evidence} />,
  );
  await expect
    .element(screen.getByText(/offline by design/i).first())
    .toBeVisible();
  await expect
    .element(screen.getByTestId("benchmarks-ref"))
    .toHaveTextContent("docs/BENCHMARKS.md");
  expect(screen.container.querySelector('[data-testid="demo-video-link"]')).toBeNull();
});

test("renders the demo-video link when the evidence carries it", async () => {
  const screen = await render(
    <OfflineNotice
      detail={OFF_STATUS.detail}
      evidence={{
        benchmarks: "docs/BENCHMARKS.md",
        replay: "/api/replay/GS-08a",
        demo_video: "https://example.com/releases/demo.mp4",
      }}
    />,
  );
  await expect
    .element(screen.getByTestId("demo-video-link"))
    .toHaveAttribute("href", "https://example.com/releases/demo.mp4");
});
