"""P6 task 5 — ``test_ui_attribution_obligations``: the LICENSING.md attribution
obligations as an executable Tier-1 check, not a checklist item (P6_SPEC §4).

The rendered chrome lives in ``ui/app/src/components/Footer.tsx`` (asserted rendered
+ measured in Footer.test.tsx, vitest browser mode); THIS test pins the committed
source of truth so removing any obligation string or the logo asset fails CI even
without a node toolchain.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_SRC = REPO_ROOT / "ui" / "app" / "src"
FOOTER = UI_SRC / "components" / "Footer.tsx"
TMDB_LOGO = UI_SRC / "assets" / "tmdb.svg"
CITATIONS = UI_SRC / "lib" / "citations.ts"
LICENSING = REPO_ROOT / "docs" / "LICENSING.md"

# The exact TMDB FAQ notice — verbatim, including the final period.
TMDB_NOTICE = "This product uses the TMDB API but is not endorsed or certified by TMDB."
# The exact IMDb courtesy line from the non-commercial dataset terms.
IMDB_COURTESY = "Information courtesy of IMDb (https://www.imdb.com). Used with permission."


def test_ui_attribution_obligations() -> None:
    chrome = FOOTER.read_text(encoding="utf-8")
    assert TMDB_NOTICE in chrome, "exact TMDB FAQ notice missing from the rendered chrome"
    assert IMDB_COURTESY in chrome, "IMDb courtesy line missing from the rendered chrome"
    assert "CC BY-SA 4.0" in chrome, "Wikipedia license label missing from the rendered chrome"
    assert "non-commercial" in chrome, "IMDb non-commercial-demo note missing"
    assert "tmdb.svg" in chrome, "the TMDB logo asset is not referenced by the chrome"


def test_tmdb_logo_is_a_real_committed_svg() -> None:
    assert TMDB_LOGO.exists(), "official TMDB logo asset missing (ui/app/src/assets/tmdb.svg)"
    head = TMDB_LOGO.read_text(encoding="utf-8")[:200]
    assert head.lstrip().startswith("<svg"), "tmdb.svg is not an SVG document"
    assert TMDB_LOGO.stat().st_size > 500, "suspiciously small — not the official mark?"


def test_tmdb_logo_prominence_rule_is_pinned_in_css() -> None:
    """The FAQ condition (logo less prominent than our mark) is enforced two ways:
    measured in the browser (Footer.test.tsx) and pinned here against silent CSS edits."""
    css = (UI_SRC / "styles.css").read_text(encoding="utf-8")
    assert ".tmdb-logo" in css and "height: 0.8rem" in css
    assert "font-size: 1.9rem" in css  # the Sutradhar mark stays the primary mark


def test_wikipedia_citations_are_revision_pinned() -> None:
    """Per-claim links carry ?oldid= (the CC BY-SA link-back, made revision-exact)."""
    builders = CITATIONS.read_text(encoding="utf-8")
    assert "oldid=" in builders
    assert "creativecommons.org/licenses/by-sa/4.0" in FOOTER.read_text(encoding="utf-8")


def test_licensing_doc_records_the_ui_obligations() -> None:
    """LICENSING.md carries the two P6 rows this task discharges (spec §1.1 gaps)."""
    licensing = LICENSING.read_text(encoding="utf-8")
    assert "less prominent" in licensing, "TMDB logo-prominence clause not documented"
    assert IMDB_COURTESY in licensing, "IMDb courtesy line not documented"
