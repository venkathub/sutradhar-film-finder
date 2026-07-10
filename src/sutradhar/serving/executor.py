"""Live tool executor for the P5 API path (task 7, P5_SPEC §2.1/§2.3).

Mirrors the eval driver's ``build_executor`` — same five v0 tools, same
validated-call-only contract — with ONE difference: ``search_by_plot`` runs the real
hybrid ``Retriever`` (winner config, live HTTP providers) instead of the recorded P2
replay. The P2 promise holds: ``repository.search_by_plot`` and the retrieval pipeline
are unchanged; only the providers swapped.

Degradation judgment call (flagged per plan): a *provider* outage (embedding/rerank
sidecar off) is narrower than an *LLM* outage — the graph tools (resolve_title,
get_work, get_versions, refine_filter) are DB-only and still work. So
``ProviderUnavailableError`` becomes a ``ToolExecutionError`` fed back to the model
("plot search unavailable"), letting it answer via the title/graph channel rather than
aborting the whole turn. The LLM being off still aborts the turn (orchestrator).
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from sutradhar.evals.driver import ToolExecutionError, ToolExecutor
from sutradhar.graph import repository
from sutradhar.rag.providers import ProviderUnavailableError
from sutradhar.rag.retrieve import Retriever


def build_live_executor(session: Session, retriever: Retriever) -> ToolExecutor:
    """Maps validated calls onto the five v0 repository functions (live retrieval leg)."""

    def _uuid(value: Any, field: str) -> uuid.UUID:
        try:
            return uuid.UUID(str(value))
        except ValueError as exc:
            raise ToolExecutionError(f"{field}: {value!r} is not a known id") from exc

    def execute(tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool == "resolve_title":
            return repository.resolve_title(
                session, args["title"], args.get("language")
            ).model_dump(mode="json")
        if tool == "search_by_plot":
            try:
                return repository.search_by_plot(
                    session,
                    args["description"],
                    int(args.get("top_k", 10)),
                    retriever=retriever,
                ).model_dump(mode="json")
            except ProviderUnavailableError as exc:
                raise ToolExecutionError(
                    f"plot search unavailable — {exc.provider} endpoint {exc.status} "
                    "(title/graph tools still work)"
                ) from exc
        if tool == "get_work":
            result = repository.get_work(session, _uuid(args["work_id"], "work_id"))
            if result is None:
                raise ToolExecutionError(f"work_id {args['work_id']!r} not found")
            return result.model_dump(mode="json")
        if tool == "get_versions":
            return repository.get_versions(
                session,
                _uuid(args["work_id"], "work_id"),
                scope=args.get("scope", "indian"),
                include_sequels=bool(args.get("include_sequels", False)),
            ).model_dump(mode="json")
        if tool == "refine_filter":
            version_set = [_uuid(v, "version_set") for v in args["version_set"]]
            by = repository.RefineBy.model_validate(args["by"])
            return repository.refine_filter(session, version_set, by).model_dump(mode="json")
        raise ToolExecutionError(f"no executor for tool {tool!r}")  # pragma: no cover

    return execute
