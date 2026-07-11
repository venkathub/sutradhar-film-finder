// App shell (P6 task 2). The chat panel, degradation states, version cards,
// citations, and trace view land in tasks 3–6 — this scaffold pins the layout
// regions and the generated-artifact posture (tool labels imported, never typed).
import toolLabels from "./generated/tool_labels.json";

export default function App() {
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
      </header>
      <main className="app-main">
        <p className="placeholder" data-testid="scaffold-placeholder">
          Chat interface lands here (P6 tasks 3–6). Trace view renders the{" "}
          {Object.keys(toolLabels.tools).length} tools of schema{" "}
          {toolLabels.schema_version}.
        </p>
      </main>
    </div>
  );
}
