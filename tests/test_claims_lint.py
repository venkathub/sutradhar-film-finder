"""P7 task 13 (DEC-P7-1) — claims-lint tripwire.

Locks the P7 doc-truth reconciliation (tasks 9–11) in place: retired claims
must not be re-introduced into the standing docs. Deliberately narrow patterns
(exact retired phrasings, per-file scoping, explicit allowlisted contexts) so
the tripwire cannot false-positive on the honest corrections themselves.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The standing claim surface: what an interviewer reads first.
STANDING_DOCS = (
    "README.md",
    "CLAUDE.md",
    "docs/ROADMAP.md",
    "docs/PORTFOLIO.md",
    "docs/BENCHMARKS.md",
    "docs/RUNBOOK.md",
)

# (pattern, files, description, allowed_context_regexes)
# A match is a violation UNLESS one allowed-context regex matches the same line.
RETIRED_CLAIMS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    (
        r"sub-2[\s-]?min",
        STANDING_DOCS,
        "retired timing claim — measured posture is ~545 s ephemeral create (RUNBOOK)",
        (r"NOT a sub-2-min", r"not a sub-2-min", r'"sub-2-min resume" claim'),
    ),
    (
        r"\$12[–-]17",
        STANDING_DOCS,
        "retired estimate-based project total — actuals recomputed in PORTFOLIO (P7 task 10)",
        (r"estimate", r"retired"),
    ),
    (
        # Present-tense victory claim; the honest form is past-tense rule + CUT verdict.
        r"QLoRA (measurably )?(beat|beats) the (well-prompted )?base",
        STANDING_DOCS,
        "the FT verdict is CUT (DEC-P4-9) — no victory claim may stand",
        (r"does not measurably beat", r"settled", r"CUT", r"rewritten"),
    ),
    (
        r"0 hallucinated|hallucinated-movie rate (of )?(0|zero)",
        ("README.md", "docs/PORTFOLIO.md"),
        "zero-hallucination claims must carry the two-layer framing (model GS-02 = 1 ⚠)",
        (r"GS-02\s*=\s*1", r"served.layer", r"served-layer", r"output gate", r"dry-run"),
    ),
    (
        # PR #9 blocking finding 3 (+ residual nit): the deliverable is intra-rater
        # test-retest (DEC-P7-6); neither a human–human ceiling, a second-annotator
        # report, nor a "closed" single-annotator loop may be claimed as existing.
        r"human.human κ ceiling|human.human kappa ceiling|second.annotator report"
        r"|clos(ed|ing) the single.annotator",
        STANDING_DOCS,
        "DEC-P7-6: intra-rater test-retest only — bounds, never closes; no ceiling exists yet",
        (r"remains the (additive )?upgrade path", r"NOT presented", r"intra-rater", r"DEC-P7-6"),
    ),
)


def _lines(rel_path: str) -> list[str]:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8").splitlines()


def test_retired_claims_stay_retired() -> None:
    violations: list[str] = []
    for pattern, files, description, allowed in RETIRED_CLAIMS:
        regex = re.compile(pattern, re.IGNORECASE)
        allowed_regexes = [re.compile(a, re.IGNORECASE) for a in allowed]
        for rel_path in files:
            for lineno, line in enumerate(_lines(rel_path), start=1):
                if not regex.search(line):
                    continue
                if any(a.search(line) for a in allowed_regexes):
                    continue
                violations.append(f"{rel_path}:{lineno}: [{description}] {line.strip()[:120]}")
    assert not violations, "retired claims re-introduced:\n" + "\n".join(violations)


def test_ft_verdict_is_stated_in_the_flagship_docs() -> None:
    """The CUT verdict must remain visible where the FT story is told."""
    for rel_path in ("README.md", "CLAUDE.md"):
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        assert re.search(r"CUT.*DEC-P4-9|DEC-P4-9.*CUT", text, re.IGNORECASE | re.DOTALL), (
            f"{rel_path} lost the settled FT verdict (CUT, DEC-P4-9)"
        )


def test_benchmarks_two_layer_annotation_present() -> None:
    text = (REPO_ROOT / "docs/BENCHMARKS.md").read_text(encoding="utf-8")
    assert "Two-layer hallucination framing" in text, (
        "BENCHMARKS.md lost the P7 two-layer annotation under Table 2"
    )
