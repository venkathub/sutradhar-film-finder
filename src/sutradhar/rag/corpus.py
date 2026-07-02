"""Gate-visible corpus builder (P2_SPEC §2.2): plot chunks + metadata cards → ``chunks``.

Reads ONLY the verification-gate views (``ground_truth_versions`` ⋈ ``plot_texts``), so
CANDIDATE edges and conflict-hidden records are excluded **by construction** — they never
reach the retrieval index. Two document kinds per Version:

1. **Plot chunks** — Wikipedia plot prose primary, TMDB overview fill; recursive
   paragraph-boundary chunks (DEC-P2-3), each prefixed with a metadata header
   (``"{title} ({year}, {language}) — remake of {original} …. "``) so every embedded
   unit carries its remake/dub lineage into dense space.
2. **Metadata card** — one synthetic doc per Version (title/AKAs/language/year/lead
   cast/director/relationship): the dense target for cast-anchored + title-ish queries.

Native-script plots are chunked as-is (BGE-M3 is multilingual); headers stay English +
native title (the AKA lines on the card are the cross-lingual anchor).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from sutradhar.graph.repository import gt_edges, gt_versions
from sutradhar.graph.schema import Chunk, Person, PlotText, VersionCast, VersionTitle
from sutradhar.rag.chunking import (
    CHUNK_CONFIGS,
    CHUNKER_NAME,
    ChunkConfig,
    chunk_text,
    content_hash,
)

# Graph-derived card provenance: Wikidata spine is CC0; cast/AKA enrichment carries the
# TMDB attribution requirement (docs/LICENSING.md).
CARD_LICENSE = "CC0 (Wikidata) + TMDB attribution"

LANGUAGE_NAMES = {
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "kn": "Kannada",
    "bn": "Bengali",
    "si": "Sinhala",
    "zh": "Chinese",
    "en": "English",
}

RELATIONSHIP_WORDS = {
    "is_remake_of": "remake of",
    "is_official_dub_of": "official dub of",
    "is_unofficial_remake_of": "unofficial remake of",
}


def language_name(code: str | None) -> str:
    if code is None:
        return "unknown language"
    return LANGUAGE_NAMES.get(code, code)


@dataclass(frozen=True)
class VersionMeta:
    """Everything the header/card needs about one gate-visible Version."""

    version_id: uuid.UUID
    work_id: uuid.UUID
    title: str
    language: str
    year: int | None
    is_original: bool
    relationship: str | None  # outgoing gate-visible remake/dub edge type
    original_title: str | None  # dst of that edge
    original_language: str | None
    original_year: int | None
    aka_titles: tuple[str, ...]
    leads: tuple[str, ...]
    directors: tuple[str, ...]


def build_header(meta: VersionMeta) -> str:
    """P2_SPEC §2.2 header — what carries Papanasam's Drishyam lineage into dense space."""
    year = f", {meta.year}" if meta.year is not None else ""
    header = f"{meta.title} ({language_name(meta.language)}{year})"
    if meta.relationship and meta.original_title:
        orig_year = f", {meta.original_year}" if meta.original_year is not None else ""
        header += (
            f" — {RELATIONSHIP_WORDS[meta.relationship]} {meta.original_title}"
            f" ({language_name(meta.original_language)}{orig_year})"
        )
    return header + ". "


def metadata_card_text(meta: VersionMeta) -> str:
    """One synthetic document per Version: the cast/title-anchored dense target."""
    parts = [build_header(meta).strip()]
    if meta.is_original:
        parts.append("This is the original version of this story.")
    if meta.aka_titles:
        parts.append("Also known as: " + "; ".join(meta.aka_titles) + ".")
    if meta.directors:
        parts.append("Directed by " + ", ".join(meta.directors) + ".")
    if meta.leads:
        parts.append("Starring " + ", ".join(meta.leads) + ".")
    parts.append(f"A {language_name(meta.language)} film.")
    return " ".join(parts)


class CorpusReport(BaseModel):
    configs: list[str]
    versions_seen: int = 0
    cards_written: int = 0
    plot_docs: int = 0
    plot_chunks: dict[str, int] = Field(default_factory=dict)  # per chunk_config
    versions_without_plots: list[str] = Field(default_factory=list)


def _cast_names(session: Session, role_kind: str) -> dict[uuid.UUID, list[str]]:
    rows = session.execute(
        select(VersionCast.version_id, Person.name, VersionCast.billing_order)
        .join(Person, Person.person_id == VersionCast.person_id)
        .where(VersionCast.role_kind == role_kind)
        .order_by(VersionCast.billing_order, Person.name)
    ).all()
    names: dict[uuid.UUID, list[str]] = {}
    for version_id, name, _ in rows:
        names.setdefault(version_id, []).append(name)
    return names


def _aka_titles(session: Session) -> dict[uuid.UUID, list[str]]:
    rows = session.execute(select(VersionTitle.version_id, VersionTitle.title)).all()
    akas: dict[uuid.UUID, set[str]] = {}
    for version_id, title in rows:
        akas.setdefault(version_id, set()).add(title)
    return {vid: sorted(titles) for vid, titles in akas.items()}


def load_version_metas(session: Session) -> list[VersionMeta]:
    """Gate-visible versions + lineage (outgoing remake/dub edge → dst version) + card facts."""
    version_rows = session.execute(select(gt_versions).order_by(gt_versions.c.title)).all()
    by_id = {r.version_id: r for r in version_rows}
    edge_rows = session.execute(
        select(gt_edges.c.src_id, gt_edges.c.edge_type, gt_edges.c.dst_id).where(
            gt_edges.c.edge_type.in_(tuple(RELATIONSHIP_WORDS))
        )
    ).all()
    lineage = {r.src_id: r for r in edge_rows}
    leads = _cast_names(session, "lead")
    directors = _cast_names(session, "director")
    akas = _aka_titles(session)

    metas: list[VersionMeta] = []
    for row in version_rows:
        edge = lineage.get(row.version_id)
        # dst must itself be gate-visible to be quotable in a header (view-join semantics).
        dst = by_id.get(edge.dst_id) if edge is not None else None
        metas.append(
            VersionMeta(
                version_id=row.version_id,
                work_id=row.work_id,
                title=row.title,
                language=row.language,
                year=row.release_year,
                is_original=row.is_original,
                relationship=edge.edge_type if edge is not None and dst is not None else None,
                original_title=dst.title if dst is not None else None,
                original_language=dst.language if dst is not None else None,
                original_year=dst.release_year if dst is not None else None,
                aka_titles=tuple(t for t in akas.get(row.version_id, []) if t != row.title),
                leads=tuple(leads.get(row.version_id, [])[:5]),
                directors=tuple(directors.get(row.version_id, [])),
            )
        )
    return metas


def _plot_docs(session: Session, version_ids: set[uuid.UUID]) -> dict[uuid.UUID, list[PlotText]]:
    """Plot rows per gate-visible version: Wikipedia primary, TMDB overview fill."""
    rows = session.scalars(
        select(PlotText).where(PlotText.version_id.in_(version_ids))
    ).all()
    by_version: dict[uuid.UUID, list[PlotText]] = {}
    for row in rows:
        by_version.setdefault(row.version_id, []).append(row)
    docs: dict[uuid.UUID, list[PlotText]] = {}
    for version_id, plots in by_version.items():
        wikipedia = [p for p in plots if p.source == "wikipedia"]
        chosen = wikipedia if wikipedia else plots  # TMDB fills only when Wikipedia is absent
        # Deterministic doc order (uuid PKs are not stable across rebuilds).
        docs[version_id] = sorted(
            chosen, key=lambda p: (p.source, p.language or "", p.revision_id or "")
        )
    return docs


def build_corpus(
    session: Session, configs: tuple[ChunkConfig, ...] = CHUNK_CONFIGS
) -> CorpusReport:
    """Rebuild ``chunks`` for every config: idempotent delete-and-reinsert per config key."""
    report = CorpusReport(configs=[c.name for c in configs])
    metas = load_version_metas(session)
    report.versions_seen = len(metas)
    docs = _plot_docs(session, {m.version_id for m in metas})
    report.plot_docs = sum(len(d) for d in docs.values())

    for config in configs:
        session.query(Chunk).filter(
            Chunk.chunker == CHUNKER_NAME, Chunk.chunk_config == config.name
        ).delete(synchronize_session=False)
        report.plot_chunks[config.name] = 0
        for meta in metas:
            header = build_header(meta)
            seq = 0
            for plot in docs.get(meta.version_id, []):
                for body in chunk_text(plot.text, config):
                    text = header + body
                    session.add(
                        Chunk(
                            version_id=meta.version_id,
                            work_id=meta.work_id,
                            plot_id=plot.plot_id,
                            kind="plot",
                            seq=seq,
                            text=text,
                            language=plot.language,
                            chunker=CHUNKER_NAME,
                            chunk_config=config.name,
                            content_hash=content_hash(text),
                            license=plot.license,
                        )
                    )
                    seq += 1
                    report.plot_chunks[config.name] += 1
            card = metadata_card_text(meta)
            session.add(
                Chunk(
                    version_id=meta.version_id,
                    work_id=meta.work_id,
                    plot_id=None,
                    kind="metadata_card",
                    seq=0,
                    text=card,
                    language="en",
                    chunker=CHUNKER_NAME,
                    chunk_config=config.name,
                    content_hash=content_hash(card),
                    license=CARD_LICENSE,
                )
            )
            report.cards_written += 1
            if not docs.get(meta.version_id) and config is configs[0]:
                report.versions_without_plots.append(f"{meta.title} ({meta.language})")
    session.flush()
    return report
