"""The ``DATA_SOURCES.md`` per-field precedence table as code (P1_SPEC §1.5).

Each field row maps to a strategy; disagreements are **never silent**:

- Rule-decidable disagreements return a ``resolved`` conflict record (both values + sources
  preserved, ``resolution={"by": "rule", ...}``) — recorded, but the row stays live.
- Rule-*un*decidable disagreements (e.g. a release-year split with no majority) return an
  ``open`` conflict — the gate views hide the row until a human resolves it.

Strategies (mapping the table's "Conflict rule" column):

- ``hub`` (external IDs): Wikidata is the linking hub; its value wins, HIGH.
- ``primary_corroborate`` (canonical title, original language, director, lead cast):
  primary source wins; HIGH when a second source agrees; disagreement → resolved-by-rule.
- ``majority`` (release year): all agree → HIGH; majority → resolved-by-rule;
  split → OPEN conflict.
- ``union`` (AKA/dub titles): multi-valued; union + dedupe; appears-in-≥2 → HIGH per value.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

Strategy = Literal["hub", "primary_corroborate", "majority", "union"]

# Field → (strategy, primary-source order). Mirrors docs/DATA_SOURCES.md §"Per-field precedence".
FIELD_STRATEGIES: dict[str, tuple[Strategy, tuple[str, ...]]] = {
    "external_id": ("hub", ("wikidata",)),
    "canonical_title": ("primary_corroborate", ("tmdb", "wikidata", "wikipedia")),
    "aka_title": ("union", ()),
    "release_year": ("majority", ("tmdb", "wikidata", "imdb", "human")),
    "original_language": ("primary_corroborate", ("tmdb", "wikidata", "imdb")),
    "director": ("primary_corroborate", ("tmdb", "wikidata", "imdb")),
    "lead_cast": ("primary_corroborate", ("tmdb", "wikipedia", "imdb")),
}


@dataclass(frozen=True)
class Observation:
    """One source's claim about a field value."""

    value: Any
    source: str  # wikidata | tmdb | imdb | wikipedia | human


@dataclass
class Resolution:
    """Outcome of applying a field's precedence rule to its observations."""

    value: Any
    confidence: Literal["HIGH", "MEDIUM"]
    conflict: Literal["none", "resolved", "open"] = "none"
    conflict_values: list[dict[str, Any]] = field(default_factory=list)
    resolution: dict[str, Any] | None = None


def _conflict_values(observations: list[Observation]) -> list[dict[str, Any]]:
    return [{"value": o.value, "source": o.source} for o in observations]


def resolve_field(field_name: str, observations: list[Observation]) -> Resolution:
    """Apply the precedence table to one field's observations (≥1 required)."""
    if not observations:
        raise ValueError("resolve_field requires at least one observation")
    strategy, primary_order = FIELD_STRATEGIES[field_name]

    if strategy == "union":
        raise ValueError("union fields are multi-valued: use union_values()")

    values = [o.value for o in observations]
    distinct = {repr(v) for v in values}
    agree = len(distinct) == 1

    if strategy == "hub":
        hub_obs = next((o for o in observations if o.source in primary_order), None)
        chosen = hub_obs or observations[0]
        if agree or len(observations) == 1:
            return Resolution(chosen.value, "HIGH" if hub_obs else "MEDIUM")
        return Resolution(
            chosen.value,
            "HIGH" if hub_obs else "MEDIUM",
            conflict="resolved",
            conflict_values=_conflict_values(observations),
            resolution={"by": "rule", "rule": "hub:wikidata", "chosen_value": chosen.value},
        )

    if strategy == "primary_corroborate":
        primary_obs = next(
            (o for src in primary_order for o in observations if o.source == src),
            observations[0],
        )
        if agree:
            return Resolution(primary_obs.value, "HIGH" if len(observations) >= 2 else "MEDIUM")
        corroborated = sum(1 for o in observations if o.value == primary_obs.value) >= 2
        return Resolution(
            primary_obs.value,
            "HIGH" if corroborated else "MEDIUM",
            conflict="resolved",
            conflict_values=_conflict_values(observations),
            resolution={
                "by": "rule",
                "rule": f"primary:{primary_obs.source}",
                "chosen_value": primary_obs.value,
            },
        )

    # majority (release year): all agree → HIGH; strict majority → resolved; split → OPEN.
    counts = Counter(repr(o.value) for o in observations)
    top_repr, top_count = counts.most_common(1)[0]
    top_value = next(o.value for o in observations if repr(o.value) == top_repr)
    if agree:
        return Resolution(top_value, "HIGH" if len(observations) >= 2 else "MEDIUM")
    if top_count * 2 > len(observations):  # strict majority
        return Resolution(
            top_value,
            "MEDIUM",  # contested value: flagged even when the rule decides
            conflict="resolved",
            conflict_values=_conflict_values(observations),
            resolution={"by": "rule", "rule": "majority", "chosen_value": top_value},
        )
    return Resolution(
        top_value,
        "MEDIUM",
        conflict="open",  # rule-undecidable: human must resolve; gate views hide the row
        conflict_values=_conflict_values(observations),
    )


def union_values(observations: list[Observation]) -> list[tuple[Any, Literal["HIGH", "MEDIUM"]]]:
    """AKA/dub titles: union + dedupe; a value appearing in ≥2 sources is HIGH."""
    by_value: dict[Any, set[str]] = {}
    order: list[Any] = []
    for o in observations:
        if o.value not in by_value:
            by_value[o.value] = set()
            order.append(o.value)
        by_value[o.value].add(o.source)
    return [(v, "HIGH" if len(by_value[v]) >= 2 else "MEDIUM") for v in order]
