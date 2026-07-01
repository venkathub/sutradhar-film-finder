# ui

Find-a-movie chat interface with citations and a trace view.

## Planned architecture
- Chat UI that shows **all** language versions of a matched film with the original clearly flagged.
- Renders per-claim citations and a trace view (retrieval + tool calls + grounding).
- Talks to the FastAPI serving layer; no neural model runs in the UI.

## Status
**Not built until P6.** P0 creates this directory as a stub only. (Frontend assets — no
`sutradhar.*` import package.)
