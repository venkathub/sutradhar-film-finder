"""Named golden GENERATION regressions — the Tier-1 CI gate for the generation surface
(P3 task 12; P3_SPEC §2.7, DEC-P2-6 posture).

Every deterministic metric is RECOMPUTED from the committed run artifact's recorded
transcripts with the same scorer bytes as the harness (``score_fixture`` /
``aggregate_metrics``), then asserted equal to the artifact's own metrics block — the
committed numbers can never drift from what the code computes. Hard gates asserted on top:
GS-02 = 0 inventions; every invalid emitted call flagged and accounted; the two seeded
faults visibly caught. Judge/RAGAS fields are checked for presence/shape only — Tier-1
never calls a model (no DB, no GPU, no network).

Pin: ``GENERATION_RUN`` env (via Settings); default = the latest committed artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sutradhar.config import Settings
from sutradhar.evals.generation_run import (
    GenerationRunArtifact,
    aggregate_metrics,
    load_generation_run,
    score_fixture,
    select_generation_fixtures,
)
from sutradhar.evals.golden import GoldenFixture, load_fixtures

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "evals" / "generation_runs"
GOLDEN_DIR = REPO_ROOT / "evals" / "golden"


@pytest.fixture(scope="module")
def artifact() -> GenerationRunArtifact:
    return load_generation_run(RUNS_DIR, Settings(_env_file=None).generation_run or None)


@pytest.fixture(scope="module")
def fixtures() -> dict[str, GoldenFixture]:
    return {f.id: f for f in load_fixtures(GOLDEN_DIR)}


# --- Recomputation: recorded metrics == what the scorer bytes compute today ---


def test_recorded_metrics_match_recomputation(
    artifact: GenerationRunArtifact, fixtures: dict[str, GoldenFixture]
) -> None:
    """THE drift gate: rescore every recorded transcript; the aggregate must equal the
    committed metrics block field-for-field (deterministic fields only — judge/RAGAS
    supplementary fields are pass-through, not recomputable without a model)."""
    rescored = [
        score_fixture(fixtures[r.fixture_id], r.transcript).model_copy(
            update={  # carry over the (non-recomputable) enrichment fields untouched
                "judge_coherence": r.judge_coherence,
                "judge_error": r.judge_error,
                "ragas": r.ragas,
            }
        )
        for r in artifact.fixtures
    ]
    recomputed = aggregate_metrics(rescored, artifact.mode)
    assert recomputed == artifact.metrics, "metrics drift: artifact != scorer recomputation"
    # And the per-fixture deterministic scores match too (finer-grained than aggregates).
    for recorded, fresh in zip(artifact.fixtures, rescored, strict=True):
        assert recorded.tool_calls == fresh.tool_calls, recorded.fixture_id
        assert recorded.intent_matches == fresh.intent_matches, recorded.fixture_id
        assert recorded.slot_counts == fresh.slot_counts, recorded.fixture_id
        assert recorded.hallucination == fresh.hallucination, recorded.fixture_id


def test_artifact_covers_the_full_generation_slice(
    artifact: GenerationRunArtifact, fixtures: dict[str, GoldenFixture]
) -> None:
    """The frozen artifact covers the FROZEN slice exactly; P7 additive fixtures
    (PENDING_CAPTURE_FIXTURES, DEC-P7-4) are excluded until the DEC-P7-7 window —
    frozen runs are never re-scored against fixtures that postdate them."""
    from sutradhar.evals.generation_run import PENDING_CAPTURE_FIXTURES

    expected_ids = {
        f.id for f in select_generation_fixtures(list(fixtures.values()))
    } - PENDING_CAPTURE_FIXTURES
    assert {r.fixture_id for r in artifact.fixtures} == expected_ids
    assert artifact.metrics.fixtures_completed == len(expected_ids)  # every conversation ended


# --- Hard gates (P3_SPEC §4 gate table) ---


def test_no_hallucinated_movie_on_gs02(artifact: GenerationRunArtifact) -> None:
    """The faithfulness gate, RELATIVE form (user-confirmed 2026-07-04): the recomputed
    GS-02 invention count must equal the pinned artifact's recorded value — no NEW
    hallucinations can land between GPU windows. The =0 TARGET is not met by either live
    P4 column (base invented 'Pushpa', QLoRA 'Salaar' — recorded in BENCHMARKS + the
    DEC-P4 verdict); it stays a hard clause inside `make ft-verdict` (DEC-P4-8) and the
    absolute assertion returns the moment a pinned column achieves 0."""
    recomputed = sum(
        len(turn.inventions)
        for result in artifact.fixtures
        if result.slice == "negative"
        for turn in result.hallucination
    )
    assert recomputed == artifact.metrics.gs02_inventions  # drift gate: byte-honest
    assert recomputed <= 1, "GS-02 inventions grew beyond the recorded P4 baseline"
    if artifact.mode == "dry_run":
        assert recomputed == 0  # the mock harness must stay clean


def test_every_invalid_emitted_call_is_flagged_and_accounted(
    artifact: GenerationRunArtifact,
) -> None:
    """Schema-validity accounting: invalid calls carry violations, were never executed,
    and the artifact's invalid counts equal the transcript-level truth."""
    for result in artifact.fixtures:
        invalid = [c for c in result.transcript.calls if not c.schema_valid]
        for call in invalid:
            assert call.validation_errors, (result.fixture_id, call.tool)
            assert call.executed is False and call.result is None
        if result.tool_calls is not None:
            assert result.tool_calls.invalid_emitted == len(invalid), result.fixture_id
            assert result.tool_calls.emitted_total == len(result.transcript.calls)


def test_seeded_faults_visible_in_the_dry_run(artifact: GenerationRunArtifact) -> None:
    """The committed dry-run must PROVE the gates catch faults (P3_SPEC §1.13):
    exactly one hallucinated tool call and exactly one invented movie, both outside GS-02."""
    if artifact.mode != "dry_run":
        pytest.skip("seeded faults are a dry-run property; live runs are clean captures")
    invalid_calls = [
        (r.fixture_id, c.tool)
        for r in artifact.fixtures
        for c in r.transcript.calls
        if not c.schema_valid
    ]
    assert invalid_calls == [("GS-07a", "lookup_movie")]
    inventions = [
        (r.fixture_id, i) for r in artifact.fixtures for t in r.hallucination for i in t.inventions
    ]
    assert inventions == [("GS-07e", "Chokher Aloy")]
    assert artifact.metrics.schema_validity is not None
    assert artifact.metrics.schema_validity < 1.0  # the caught fault is VISIBLE in Table 2 terms
    assert artifact.metrics.faithfulness is not None and artifact.metrics.faithfulness < 1.0


def test_dry_run_honesty_invariants(artifact: GenerationRunArtifact) -> None:
    if artifact.mode == "dry_run":
        assert artifact.serving is None
        assert artifact.metrics.latency_p50_ms is None
        assert artifact.metrics.tokens_per_sec is None
        assert artifact.model == "mock"


# --- Judge/RAGAS: presence/shape only (never re-judged on PRs, §2.7) ---


def test_judge_fields_shape_never_rejudged(artifact: GenerationRunArtifact) -> None:
    for result in artifact.fixtures:
        if result.judge_coherence is not None:
            assert 0.0 <= result.judge_coherence <= 1.0
            assert result.slice == "backtracking"  # coherence is a GS-08 rubric
        if result.ragas is not None:
            for value in (result.ragas.faithfulness, result.ragas.answer_relevancy):
                assert value is None or 0.0 <= value <= 1.0
    if artifact.judge is not None:
        for rubric_config in artifact.judge.values():
            assert rubric_config.get("model") and rubric_config.get("prompt_hash")


# --- Label pass-through (P3_SPEC §4 named regressions, generation layer) ---


def test_dub_labels_never_presented_as_remake(artifact: GenerationRunArtifact) -> None:
    """GS-04 semantics through generation: an answer naming a version whose tool-result
    relationship is is_official_dub_of must never call it a remake (pass-through check;
    vacuous until a dub franchise enters the generation slice, enforced from then on)."""
    for result in artifact.fixtures:
        dub_titles: set[str] = set()
        for call in result.transcript.calls:
            for version in (call.result or {}).get("versions", []) or []:
                if isinstance(version, dict) and (
                    version.get("relationship") == "is_official_dub_of"
                ):
                    dub_titles.add(str(version.get("title", "")).casefold())
        for answer in result.transcript.answers:
            if not answer:
                continue
            for line in answer.splitlines():
                lowered = line.casefold()
                if "remake" in lowered:
                    for title in dub_titles:
                        assert title not in lowered, (
                            f"{result.fixture_id}: dub {title!r} presented as remake"
                        )


# --- Stamp integrity ---


def test_stamp_pins_current_prompt_and_schema(artifact: GenerationRunArtifact) -> None:
    """The committed run must be from THIS prompt/schema/golden state — a stale artifact
    after a re-pin is a silent gate hole."""
    import hashlib

    from sutradhar.evals.generation_run import golden_set_hash
    from sutradhar.evals.prompts import load_prompt_artifacts

    assert artifact.prompt_hash == load_prompt_artifacts(REPO_ROOT / "evals/prompts").prompt_hash
    # P7 task 15 (DEC-P7-4): the golden DIR hash legitimately moved when the additive
    # pending-capture fixtures landed; the frozen artifact pins the 2026-07-04 state,
    # asserted here as the recorded constant (artifact-tampering tripwire). Drift in
    # the fixtures the artifact actually SCORED is still caught content-level by the
    # recompute tests above; the NEXT capture window re-pins the live hash.
    frozen_golden_hash = "f8fd77a0a60354d949a5baca8ebcc600028bba1ff692e36b169e4c7a11017469"
    assert artifact.stamp.golden_set_hash == frozen_golden_hash
    assert golden_set_hash(GOLDEN_DIR)  # live hash computable (re-pinned next window)
    schema_sha = hashlib.sha256(
        (REPO_ROOT / "docs/phases/tool_schema.v0.json").read_bytes()
    ).hexdigest()
    assert artifact.stamp.tool_schema_sha256 == schema_sha
    assert artifact.tool_schema_version == "v0"
    assert artifact.retrieval_run  # the DEC-P3-8 replay pin is recorded
