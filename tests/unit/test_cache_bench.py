"""Tests for the semantic-cache benchmark: QQP loading, scoring, tuning, reporting."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from groundwire.cache.embeddings import Embedder, Vector, normalise
from groundwire.cache.guards import NUMBERS_DIFFER, PLACEHOLDERS_DIFFER, semantic_mismatch
from groundwire.cache.keys import normalise_query
from groundwire.core.errors import DatasetError
from groundwire.evals.cachebench import cli
from groundwire.evals.cachebench.report import build_summary, render_markdown
from groundwire.evals.cachebench.scoring import (
    EmbeddingStats,
    PairScore,
    measure_lookup_latency,
    outcome,
    score_pairs,
    threshold_for_false_hit_budget,
)
from groundwire.evals.qqp import QuestionPair, load_qqp, qqp_path

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

ROWS = [
    ("How do I learn Python?", "What is the best way to learn Python?", 1),
    ("How do I learn Python?", "How do I learn Java?", 0),
    ("What is 2+2?", "What is 2+3?", 0),
    ("Best phone under 10000?", "best phone under 10000", 1),
    ("Why is the sky blue?", "What makes the sky look blue?", 1),
    ("What is love?", "What is hate?", 0),
]


def _write_tsv(path: Path, rows: Sequence[tuple[str, str, object]],
               label_column: str = "is_duplicate") -> Path:
    lines = [f"id\tquestion1\tquestion2\t{label_column}"]
    lines += [f"{i}\t{q1}\t{q2}\t{label}" for i, (q1, q2, label) in enumerate(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class KeywordEmbedder(Embedder):
    """Deterministic embeddings: a few keywords decide the direction."""

    name = "keyword"
    AXES = ("python", "java", "sky", "love", "hate", "phone", "2")

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.seen.extend(texts)
        out = []
        for text in texts:
            vector = [1.0 if axis in text else 0.0 for axis in self.AXES] + [0.3]
            if "learn" in text:
                vector[0] += 0.4  # "learn" pulls everything toward python: a hard negative
            out.append(normalise(vector))
        return out


def _score(similarity: float, duplicate: bool, *, exact: bool = False,
           guard: str | None = None) -> PairScore:
    pair = QuestionPair(id="x", question1="a", question2="b", is_duplicate=duplicate)
    return PairScore(pair=pair, similarity=similarity, exact=exact, guard=guard)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("incoming", "cached", "expected"), [
    ("total of invoice inv-1043", "total of invoice inv-1042", NUMBERS_DIFFER),
    ("refund [email_1]", "refund my order", PLACEHOLDERS_DIFFER),
    ("how do i reset my password", "i forgot my password, how do i change it", None),
    ("refund [EMAIL_1] now", "please refund [email_1]", None),
])
def test_semantic_mismatch(incoming: str, cached: str, expected: str | None) -> None:
    assert semantic_mismatch(incoming, cached) == expected


# --------------------------------------------------------------------------- #
# QQP loading
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label_column", ["is_duplicate", "label"])
def test_load_tsv(tmp_path: Path, label_column: str) -> None:
    path = _write_tsv(tmp_path / "validation.tsv", ROWS, label_column)
    pairs = load_qqp(path)
    assert len(pairs) == len(ROWS)
    assert pairs[0] == QuestionPair("0", ROWS[0][0], ROWS[0][1], True)


def test_load_skips_empty_questions(tmp_path: Path) -> None:
    path = _write_tsv(tmp_path / "v.tsv", [*ROWS[:2], ("", "x", 0)])
    assert len(load_qqp(path)) == 2


def test_load_sampling_is_seeded(tmp_path: Path) -> None:
    path = _write_tsv(tmp_path / "v.tsv", ROWS)
    first = load_qqp(path, limit=3, seed=1)
    assert first == load_qqp(path, limit=3, seed=1)
    assert len(first) == 3


@pytest.mark.parametrize(("rows", "message"), [
    ([("a", "b", -1)], "label must be 0 or 1"),
    ([("", "", 0)], "no usable question pairs"),
])
def test_load_rejects_bad_rows(tmp_path: Path, rows: list[tuple[str, str, object]],
                               message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        load_qqp(_write_tsv(tmp_path / "v.tsv", rows))


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="make bench-cache-data"):
        load_qqp(tmp_path / "missing.tsv")
    bad = tmp_path / "bad.tsv"
    bad.write_text("a\tb\n1\t2\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="expected columns"):
        load_qqp(bad)
    with pytest.raises(DatasetError, match="unsupported"):
        load_qqp(tmp_path / "data.csv")
    with pytest.raises(DatasetError, match="limit"):
        load_qqp(_write_tsv(tmp_path / "v.tsv", ROWS), limit=0)


def test_load_parquet(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"question1": [r[0] for r in ROWS], "question2": [r[1] for r in ROWS],
                      "label": [r[2] for r in ROWS], "idx": list(range(len(ROWS)))})
    path = tmp_path / "validation.parquet"
    pq.write_table(table, path)
    pairs = load_qqp(path)
    assert [p.is_duplicate for p in pairs] == [bool(r[2]) for r in ROWS]
    with pytest.raises(DatasetError, match="not found"):
        load_qqp(tmp_path / "train.parquet")


def test_missing_parquet_says_download_even_without_pyarrow(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", None)  # simulate pyarrow not installed
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(DatasetError, match="make bench-cache-data"):
        load_qqp(tmp_path / "validation.parquet")


def test_existing_parquet_without_pyarrow_says_install(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "validation.parquet"
    path.write_bytes(b"not really parquet")
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(DatasetError, match="needs pyarrow"):
        load_qqp(path)


def test_qqp_path_prefers_parquet(tmp_path: Path) -> None:
    assert qqp_path(tmp_path, "train") == tmp_path / "train.parquet"
    _write_tsv(tmp_path / "train.tsv", ROWS)
    assert qqp_path(tmp_path, "train") == tmp_path / "train.tsv"
    (tmp_path / "train.parquet").write_bytes(b"")
    assert qqp_path(tmp_path, "train") == tmp_path / "train.parquet"
    with pytest.raises(DatasetError, match="split"):
        qqp_path(tmp_path, "test")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _pairs() -> list[QuestionPair]:
    return [QuestionPair(str(i), q1, q2, bool(label)) for i, (q1, q2, label) in enumerate(ROWS)]


def test_score_pairs_uses_normalised_text_once_each() -> None:
    embedder = KeywordEmbedder()
    scores, stats = score_pairs(embedder, _pairs(), chunk_size=2)
    unique = {normalise_query(q) for r in ROWS for q in r[:2]}
    assert stats.unique_questions == len(unique) == len(embedder.seen)
    assert set(embedder.seen) == unique
    by_id = {s.pair.id: s for s in scores}
    assert by_id["3"].exact  # differs only in case and "?"
    assert by_id["2"].guard == NUMBERS_DIFFER
    # python vs java shares "learn": closer than two unrelated questions.
    assert by_id["1"].similarity > by_id["5"].similarity


def test_score_pairs_validates() -> None:
    class Short(KeywordEmbedder):
        def embed(self, texts: Sequence[str]) -> list[Vector]:
            return super().embed(texts)[:-1]

    with pytest.raises(ValueError, match="vectors"):
        score_pairs(Short(), _pairs())
    with pytest.raises(ValueError, match="chunk_size"):
        score_pairs(KeywordEmbedder(), _pairs(), chunk_size=0)


def test_hits_rules() -> None:
    assert _score(0.1, True, exact=True).hits(0.99)
    assert not _score(0.99, False, guard=NUMBERS_DIFFER).hits(0.5)
    assert _score(0.99, False, guard=NUMBERS_DIFFER).hits(0.5, use_guard=False)
    assert _score(0.9, True).hits(0.9) and not _score(0.89, True).hits(0.9)


def test_outcome_maps_to_cache_vocabulary() -> None:
    scores = [_score(0.95, True), _score(0.95, False), _score(0.5, True), _score(0.5, False)]
    m = outcome(scores, 0.9)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 1)


def test_measure_lookup_latency() -> None:
    timings = measure_lookup_latency(KeywordEmbedder(), ["a", "b", "c"], samples=2)
    assert len(timings) == 2 and all(t >= 0 for t in timings)


# --------------------------------------------------------------------------- #
# Threshold tuning
# --------------------------------------------------------------------------- #


def test_tuning_picks_lowest_threshold_within_budget() -> None:
    scores = [_score(0.99, True), _score(0.97, True), _score(0.95, False),
              _score(0.93, True), _score(0.90, False), _score(0.80, True),
              *[_score(0.1, False) for _ in range(8)]]  # 10 negatives in total
    tuned = threshold_for_false_hit_budget(scores, 0.1)  # at most 1 wrong hit
    assert tuned.threshold == 0.93
    assert tuned.hit_rate == 0.75 and tuned.false_hit_rate == 0.1
    strict = threshold_for_false_hit_budget(scores, 0.0)
    assert strict.threshold == 0.97 and strict.hit_rate == 0.5


def test_tuning_decides_ties_together() -> None:
    scores = [_score(0.95, True), _score(0.9, True), _score(0.9, False), _score(0.1, False)]
    assert threshold_for_false_hit_budget(scores, 0.0).threshold == 0.95


def test_tuning_respects_guards() -> None:
    scores = [_score(0.99, False, guard=NUMBERS_DIFFER), _score(0.9, True), _score(0.1, False)]
    assert threshold_for_false_hit_budget(scores, 0.0).threshold == 0.9
    assert threshold_for_false_hit_budget(scores, 0.0, use_guard=False).threshold == float("inf")


def test_tuning_turns_semantic_off_when_exact_matches_already_exceed_budget() -> None:
    scores = [_score(1.0, False, exact=True), _score(0.9, True), _score(0.1, False)]
    tuned = threshold_for_false_hit_budget(scores, 0.0)
    assert tuned.threshold == float("inf") and tuned.false_hit_rate == 0.5


@pytest.mark.parametrize(("scores", "budget"), [
    ([_score(0.9, True)], 0.1),
    ([_score(0.9, True), _score(0.1, False)], 1.5),
])
def test_tuning_validates(scores: list[PairScore], budget: float) -> None:
    with pytest.raises(ValueError):
        threshold_for_false_hit_budget(scores, budget)


# --------------------------------------------------------------------------- #
# Report and CLI
# --------------------------------------------------------------------------- #


def test_summary_and_markdown() -> None:
    scores, stats = score_pairs(KeywordEmbedder(), _pairs())
    tuned = [threshold_for_false_hit_budget(scores, 0.0)]
    summary = build_summary(scores, tuned=tuned, configured_threshold=0.92, stats=stats,
                            latency_ms=[1.0, 2.0, 3.0],
                            run_info={"model": "keyword", "seed": 1, "tune_pairs": 6,
                                      "date": "2026-10-20"})
    assert summary["pairs"] == 6 and summary["exact_matches"] == 1
    assert summary["guard"]["by_reason"] == {NUMBERS_DIFFER: 1}
    assert summary["guard_saves"][0]["id"] == "2"
    assert {"hit_rate", "false_hit_rate", "precision"} <= set(
        summary["configured"]["with_guard"])
    json.dumps(summary)  # serialisable
    text = render_markdown(summary)
    assert "Thresholds tuned on train" in text and "Hardest wrong matches" in text
    assert "| 0.92 |" in text


def test_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        build_summary([], tuned=[], configured_threshold=0.9, stats=EmbeddingStats(0, 0.0),
                      latency_ms=[], run_info={})


def test_markdown_shows_off_when_semantic_matching_cannot_meet_budget() -> None:
    scores = [_score(1.0, False, exact=True), _score(0.9, True), _score(0.1, False)]
    summary = build_summary(scores, tuned=[threshold_for_false_hit_budget(scores, 0.0)],
                            configured_threshold=0.92, stats=EmbeddingStats(3, 1.0),
                            latency_ms=[1.0], run_info={"model": "m", "seed": 1,
                                                        "tune_pairs": 3, "date": "d"})
    assert summary["tuned"][0]["threshold"] is None
    assert "| off |" in render_markdown(summary)


def test_cli_end_to_end(tmp_path: Path) -> None:
    data = tmp_path / "qqp"
    data.mkdir()
    _write_tsv(data / "train.tsv", ROWS * 3)
    _write_tsv(data / "validation.tsv", ROWS)
    args = cli.parse_args(["--data-dir", str(data), "--out-dir", str(tmp_path / "out"),
                           "--tune-pairs", "12", "--budgets", "0.2", "0"])
    out_dir = cli.run(args, embedder=KeywordEmbedder())
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert [t["budget"] for t in summary["tuned"]] == [0.2, 0.0]
    assert (out_dir / "report.md").read_text(encoding="utf-8").startswith("# Groundwire")
    lines = (out_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(ROWS)


def test_cli_fails_fast_on_missing_data(tmp_path: Path,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")]) == 1
    captured = capsys.readouterr()
    assert "make bench-cache-data" in captured.out + captured.err


@pytest.mark.parametrize("argv", [["--budgets", "2"], ["--tune-pairs", "-1"],
                                  ["--batch-size", "0"], ["--budgets", "x"]])
def test_cli_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(argv)
