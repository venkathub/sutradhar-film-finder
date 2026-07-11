"""P6 task 9 — the static always-available surface (DEC-P6-3).

Gates: the generator builds; the benchmark page is GENERATED from BENCHMARKS.md
(single source of truth — no metric literal in the generator); required evidence
assets present; every internal link resolves (a dead evidence link is a CI
failure, not a surprise during an interview); demo-video link present iff set.
"""

from __future__ import annotations

import importlib.util
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "site" / "generate.py"

# `site` is a stdlib module name — load the generator by file path, not import path.
_spec = importlib.util.spec_from_file_location("sutradhar_site_generate", GENERATOR)
assert _spec is not None and _spec.loader is not None
_generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generate)
BENCHMARKS_MD: Path = _generate.BENCHMARKS_MD
EVIDENCE_ASSETS: dict[str, Path] = _generate.EVIDENCE_ASSETS
build = _generate.build


@pytest.fixture(scope="module")
def dist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build(tmp_path_factory.mktemp("site") / "dist")


class _LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.internal: list[str] = []
        self.external: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name not in ("href", "src") or not value:
                continue
            if value.startswith(("http://", "https://")):
                self.external.append(value)
            elif not value.startswith(("#", "mailto:")):
                self.internal.append(value)


def _links(page: Path) -> _LinkCollector:
    collector = _LinkCollector()
    collector.feed(page.read_text(encoding="utf-8"))
    return collector


def test_site_builds_with_required_pages_and_assets(dist: Path) -> None:
    for required in ("index.html", "benchmarks.html", "style.css"):
        assert (dist / required).is_file()
    for name in EVIDENCE_ASSETS:
        assert (dist / "assets" / name).is_file(), f"evidence asset {name} not copied"
    assert (dist / "assets" / "architecture.svg").stat().st_size > 1000


def test_benchmark_numbers_come_from_benchmarks_md(dist: Path) -> None:
    """The page carries the md's OWN numbers; the generator contains no metric literal."""
    md_text = BENCHMARKS_MD.read_text(encoding="utf-8")
    html = (dist / "benchmarks.html").read_text(encoding="utf-8")
    # Every metric-looking token in the md's tables appears in the rendered page.
    metric_tokens = set(re.findall(r"\b\d+\.\d{2,}\b", md_text))
    assert len(metric_tokens) >= 10, "BENCHMARKS.md unexpectedly carries few metrics"
    for token in metric_tokens:
        assert token in html, f"metric {token} from BENCHMARKS.md missing from the page"
    # And the generator itself hand-copies none of them (single source of truth).
    generator_src = GENERATOR.read_text(encoding="utf-8")
    for token in metric_tokens:
        assert token not in generator_src, f"metric literal {token} hardcoded in generate.py"
    assert "single source of truth" in html


def test_all_internal_links_resolve(dist: Path) -> None:
    for page in dist.glob("*.html"):
        for link in _links(page).internal:
            target = dist / link.split("#", 1)[0]
            assert target.is_file(), f"{page.name}: dead internal link {link!r}"


def test_external_links_are_https_wellformed(dist: Path) -> None:
    for page in dist.glob("*.html"):
        for url in _links(page).external:
            assert url.startswith("https://"), f"{page.name}: non-https external link {url!r}"


def test_demo_video_link_only_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEMO_VIDEO_URL", raising=False)
    without = build(tmp_path / "without")
    assert "Recorded demo video" not in (without / "index.html").read_text(encoding="utf-8")

    url = "https://github.com/example/sutradhar/releases/download/v1/demo.mp4"
    monkeypatch.setenv("DEMO_VIDEO_URL", url)
    with_video = build(tmp_path / "with")
    index = (with_video / "index.html").read_text(encoding="utf-8")
    assert url in index and "Recorded demo video" in index


def test_site_is_static_only(dist: Path) -> None:
    """CLAUDE.md: the surface never serves a neural model — nothing executable ships."""
    extensions = {p.suffix for p in dist.rglob("*") if p.is_file()}
    assert extensions <= {".html", ".css", ".svg", ".png"}, f"unexpected file types {extensions}"
