"""Unit tests for the IMDb akas loader (P1 task 6) — no download, no DB.

Parse/filter tests run against ``tests/fixtures/imdb/akas_sample.tsv`` (239 real rows,
slice-filtered from the 2026-07-02 stream) and small in-memory gzip streams.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

from sutradhar.pipeline.imdb import (
    AkaRow,
    filter_akas_stream,
    parse_aka_line,
    rows_from_jsonable,
    rows_to_jsonable,
)

FIXTURE = Path(__file__).parent / "fixtures" / "imdb" / "akas_sample.tsv"


def test_parse_line_nulls_and_flag() -> None:
    row = parse_aka_line("tt3417422\t1\tDrishyam\t\\N\t\\N\toriginal\t\\N\t1\n")
    assert row == AkaRow(
        titleId="tt3417422",
        ordering=1,
        title="Drishyam",
        region=None,
        language=None,
        types="original",
        attributes=None,
        isOriginalTitle=True,
    )


def test_parse_line_language_and_region() -> None:
    row = parse_aka_line("tt2631186\t5\tबाहुबली: एक शुरुआत\tIN\thi\timdbDisplay\t\\N\t0\n")
    assert row is not None
    assert (row.language, row.region, row.isOriginalTitle) == ("hi", "IN", False)


def test_parse_header_and_malformed_rejected() -> None:
    header = "titleId\tordering\ttitle\tregion\tlanguage\ttypes\tattributes\tisOriginalTitle"
    assert parse_aka_line(header) is None
    assert parse_aka_line("tt1\tonly\tthree") is None


def test_filter_stream_keeps_only_slice_tconsts() -> None:
    tsv = (
        "titleId\tordering\ttitle\tregion\tlanguage\ttypes\tattributes\tisOriginalTitle\n"
        "tt0000001\t1\tNoise\t\\N\t\\N\t\\N\t\\N\t0\n"
        "tt3417422\t1\tDrishyam\t\\N\t\\N\toriginal\t\\N\t1\n"
        "tt0000002\t1\tMore Noise\tUS\ten\t\\N\t\\N\t0\n"
    )
    stream = io.BytesIO(gzip.compress(tsv.encode("utf-8")))
    rows = filter_akas_stream(stream, {"tt3417422"})
    assert [r.titleId for r in rows] == ["tt3417422"]
    assert rows[0].isOriginalTitle


def test_fixture_parses_completely() -> None:
    """Every non-header line of the committed real capture parses (contract holds)."""
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    rows = [parse_aka_line(line) for line in lines]
    parsed = [r for r in rows if r is not None]
    assert len(parsed) == len(lines) - 1  # only the header is skipped
    assert any(r.isOriginalTitle for r in parsed)
    assert any(r.language == "hi" for r in parsed)


def test_rows_jsonable_roundtrip() -> None:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()[1:6]
    rows = [r for r in (parse_aka_line(line) for line in lines) if r is not None]
    assert rows_from_jsonable(rows_to_jsonable(rows)) == rows
