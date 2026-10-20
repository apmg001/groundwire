"""Tests for the benchmark: dataset loading, metrics, runner, report and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from groundwire.core.config import GroundingConfig
from groundwire.core.errors import DatasetError
from groundwire.evals.cli import main, parse_args
from groundwire.evals.datasets import (
    EvalExample,
    LabelPolicy,
    load_ragtruth,
    ragtruth_context,
    ragtruth_label,
)
from groundwire.evals.metrics import (
    at_threshold,
    best_f1_threshold,
    confusion,
    percentile,
    roc_auc,
    threshold_for_block_budget,
)
from groundwire.evals.report import build_summary, render_markdown, summarise
from groundwire.evals.runner import ExampleResult, evaluate
from groundwire.guardrails.grounding import GroundingVerifier
from groundwire.guardrails.scorers import LexicalScorer

# --------------------------------------------------------------------------- #
# Fixture dataset (same layout as the real RAGTruth repo)
# --------------------------------------------------------------------------- #

SOURCES: list[dict[str, Any]] = [
    {"source_id": "s1", "task_type": "QA",
     "source_info": {"question": "price?", "passages":
                     "passage 1: The refund for order 7781 is 49.99 USD.\n\n"
                     "passage 2: Refunds are issued within 5 business days."}},
    {"source_id": "s2", "task_type": "Summary",
     "source_info": "The museum opened in 1960. It holds 4,000 paintings by local artists."},
    {"source_id": "s3", "task_type": "Data2txt",
     "source_info": {"name": "Apna Kitchen", "city": "Santa Barbara",
                     "hours": {"Monday": "16:30-21:00"}, "parking": None,
                     "review_info": [{"review_stars": 5.0, "review_text": "Great food"}]}},
]

SPAN = {"text": "x", "label_type": "Evident Conflict", "implicit_true": False}
IMPLICIT = {"text": "y", "label_type": "Evident Baseless Info", "implicit_true": True}


def _responses() -> list[dict[str, Any]]:
    rows = []
    for i in range(6):
        rows.append({"id": f"qa-good-{i}", "source_id": "s1", "split": "test", "labels": [],
                     "model": "m", "response": "The refund for order 7781 is 49.99 USD."})
        rows.append({"id": f"qa-bad-{i}", "source_id": "s1", "split": "test", "labels": [SPAN],
                     "model": "m", "response": "The refund for order 7781 is 89.99 USD."})
    rows += [
        {"id": "sum-good", "source_id": "s2", "split": "test", "labels": [],
         "response": "The museum opened in 1960 and holds 4,000 paintings."},
        {"id": "sum-bad", "source_id": "s2", "split": "test", "labels": [SPAN],
         "response": "The museum opened in 1975 and holds 9,000 paintings."},
        {"id": "d2t-good", "source_id": "s3", "split": "test", "labels": [IMPLICIT],
         "response": "Apna Kitchen is a restaurant in Santa Barbara with great food."},
        {"id": "train-1", "source_id": "s1", "split": "train", "labels": [],
         "response": "The refund for order 7781 is 49.99 USD."},
        {"id": "train-2", "source_id": "s1", "split": "train", "labels": [SPAN],
         "response": "The refund for order 7781 is 12.00 USD."},
        {"id": "train-3", "source_id": "s2", "split": "train", "labels": [],
         "response": "The museum opened in 1960."},
        {"id": "train-4", "source_id": "s3", "split": "train", "labels": [SPAN],
         "response": "Apna Kitchen opens at 9 on Monday."},
        {"id": "orphan", "source_id": "missing", "split": "test", "labels": [],
         "response": "No source for this one."},
    ]
    return rows


def _write(root: Path, sources: list[dict[str, Any]], responses: list[dict[str, Any]]) -> Path:
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "source_info.jsonl").write_text(
        "\n".join(json.dumps(s) for s in sources) + "\n", encoding="utf-8")
    (dataset / "response.jsonl").write_text(
        "\n".join(json.dumps(r) for r in responses) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def ragtruth_dir(tmp_path: Path) -> Path:
    return _write(tmp_path / "ragtruth", SOURCES, _responses())


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def test_context_per_task() -> None:
    qa = ragtruth_context("QA", SOURCES[0]["source_info"])
    assert len(qa) == 2 and "49.99" in qa[0] and "price?" not in " ".join(qa)
    assert ragtruth_context("Summary", "Some article.") == ("Some article.",)
    d2t = ragtruth_context("Data2txt", SOURCES[2]["source_info"])
    assert "name: Apna Kitchen." in d2t
    assert "hours.Monday: 16:30-21:00." in d2t
    assert "parking: not provided." in d2t
    assert "review_info 1.review_text: Great food." in d2t


@pytest.mark.parametrize(("task", "info"), [
    ("QA", "not a dict"), ("Summary", {"a": 1}), ("Data2txt", "text"), ("Poetry", "x")])
def test_context_rejects_wrong_shapes(task: str, info: Any) -> None:
    with pytest.raises(DatasetError):
        ragtruth_context(task, info)


def test_label_policies() -> None:
    assert ragtruth_label([], LabelPolicy.STRICT) is False
    assert ragtruth_label([IMPLICIT], LabelPolicy.STRICT) is True
    assert ragtruth_label([IMPLICIT], LabelPolicy.EXCLUDE_IMPLICIT_TRUE) is False
    assert ragtruth_label([IMPLICIT, SPAN], LabelPolicy.EXCLUDE_IMPLICIT_TRUE) is True


def test_load_filters_split_and_skips_orphans(ragtruth_dir: Path) -> None:
    examples = load_ragtruth(ragtruth_dir, split="test")
    ids = {e.id for e in examples}
    assert "orphan" not in ids and not any(i.startswith("train") for i in ids)
    assert len(examples) == 15
    assert [e.task for e in examples][:12] == ["QA"] * 12  # grouped by task order


def test_load_sampling_is_reproducible(ragtruth_dir: Path) -> None:
    first = load_ragtruth(ragtruth_dir, tasks=["QA"], per_task=4, seed=1)
    second = load_ragtruth(ragtruth_dir, tasks=["QA"], per_task=4, seed=1)
    assert [e.id for e in first] == [e.id for e in second]
    assert len(first) == 4


def test_load_applies_label_policy(ragtruth_dir: Path) -> None:
    examples = load_ragtruth(ragtruth_dir, tasks=["Data2txt"],
                             label_policy=LabelPolicy.EXCLUDE_IMPLICIT_TRUE)
    assert examples[0].is_hallucinated is False


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"tasks": ["Poetry"]}, "unknown tasks"),
    ({"split": "dev"}, "split"),
    ({"per_task": 0}, "per_task"),
])
def test_load_argument_errors(ragtruth_dir: Path, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        load_ragtruth(ragtruth_dir, **kwargs)


def test_load_missing_files_hint_download(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="make bench-data"):
        load_ragtruth(tmp_path)


def test_load_bad_json(tmp_path: Path) -> None:
    root = _write(tmp_path / "r", SOURCES, [])
    (root / "dataset" / "response.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_ragtruth(root)


def test_load_task_without_examples(tmp_path: Path) -> None:
    root = _write(tmp_path / "r", SOURCES, _responses()[:2])  # only QA rows
    with pytest.raises(DatasetError, match="no 'test' examples"):
        load_ragtruth(root, tasks=["Summary"])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_confusion_and_derived_metrics() -> None:
    m = confusion([True, True, False, False, True], [True, False, False, True, True])
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 1, 1)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)
    assert m.false_block_rate == pytest.approx(0.5)
    assert m.as_dict()["n"] == 5


def test_confusion_edge_cases() -> None:
    empty = confusion([], [])
    assert empty.precision == empty.recall == empty.f1 == empty.accuracy == 0.0
    with pytest.raises(ValueError, match="length"):
        confusion([True], [])


def test_threshold_rule_lower_is_suspicious() -> None:
    m = at_threshold([0.1, 0.9], [True, False], threshold=0.5)
    assert (m.tp, m.tn) == (1, 1)


def test_best_threshold_separates_perfectly() -> None:
    threshold, f1 = best_f1_threshold([0.1, 0.2, 0.8, 0.9], [True, True, False, False])
    assert f1 == 1.0
    assert 0.2 < threshold <= 0.8
    with pytest.raises(ValueError):
        best_f1_threshold([], [])


@pytest.mark.parametrize(("scores", "actual", "expected"), [
    ([0.1, 0.2, 0.8, 0.9], [True, True, False, False], 1.0),
    ([0.9, 0.8, 0.2, 0.1], [True, True, False, False], 0.0),
    ([0.5, 0.5, 0.5, 0.5], [True, False, True, False], 0.5),
    ([0.1, 0.5, 0.5, 0.9], [True, True, False, False], 0.875),
])
def test_roc_auc(scores: list[float], actual: list[bool], expected: float) -> None:
    assert roc_auc(scores, actual) == pytest.approx(expected)


def test_block_budget_threshold() -> None:
    scores = [0.05, 0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9]
    actual = [True, True, False, True, False, False, False, True]
    threshold, recall = threshold_for_block_budget(scores, actual, 0.25)
    m = at_threshold(scores, actual, threshold)
    assert m.false_block_rate <= 0.25
    assert recall == m.recall == 0.75  # catches 3 of 4 while blocking 1 of 4 good answers


def test_block_budget_zero_and_errors() -> None:
    threshold, recall = threshold_for_block_budget([0.1, 0.2], [False, True], 0.0)
    assert (threshold, recall) == (0.0, 0.0)  # cannot catch anything without blocking good
    with pytest.raises(ValueError):
        threshold_for_block_budget([0.1], [True], 1.5)
    with pytest.raises(ValueError):
        threshold_for_block_budget([], [], 0.1)


def test_roc_auc_single_class() -> None:
    assert roc_auc([0.1, 0.2], [True, True]) is None


def test_percentile() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([5, 1, 3], 0.5) == 3
    assert percentile(list(range(1, 101)), 0.95) == 95


# --------------------------------------------------------------------------- #
# Runner and report
# --------------------------------------------------------------------------- #


async def test_runner_with_lexical_scorer(ragtruth_dir: Path) -> None:
    examples = load_ragtruth(ragtruth_dir, tasks=["QA"])
    verifier = GroundingVerifier(GroundingConfig(min_supported_ratio=1.0), LexicalScorer())
    results = await evaluate(verifier, examples)
    by_id = {r.id: r for r in results}
    assert by_id["qa-good-0"].support_score == 1.0
    assert by_id["qa-bad-0"].support_score == 0.0  # invented amount
    assert by_id["qa-bad-0"].unsupported_claim == "The refund for order 7781 is 89.99 USD."


async def test_runner_handles_responses_without_claims() -> None:
    example = EvalExample("e", "QA", ("ctx",), "Sure!", False)
    verifier = GroundingVerifier(GroundingConfig(), LexicalScorer())
    [result] = await evaluate(verifier, [example])
    assert result.support_score == 1.0 and result.claims == 0


async def test_runner_records_scorer_failures() -> None:
    class Broken(LexicalScorer):
        def score_batch(self, pairs):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    example = EvalExample("e", "QA", ("The refund is 5 USD.",), "The refund is 5 USD.", True)
    [result] = await evaluate(GroundingVerifier(GroundingConfig(), Broken()), [example])
    assert result.error is not None and result.support_score == 1.0


def _result(task: str, score: float, label: bool) -> ExampleResult:
    return ExampleResult("x", task, label, score, 1, None, 1.0)


def test_summary_and_markdown() -> None:
    results = [_result("QA", 0.1, True), _result("QA", 0.9, False),
               _result("Summary", 0.2, True), _result("Summary", 0.3, False)]
    run_info = {"dataset": "RAGTruth", "split": "test", "scorer": "lexical", "nli_model": None,
                "per_task": 2, "seed": 1, "label_policy": "strict",
                "threshold_source": "tuned for best F1", "date": "2026-10-20"}
    summary = build_summary(results, 0.25, run_info)
    assert summary["overall"]["recall"] == 1.0
    assert summary["by_task"]["QA"]["f1"] == 1.0
    assert summary["overall"]["f1_flag_everything"] == pytest.approx(0.6667, abs=1e-4)
    markdown = render_markdown(summary)
    assert "| **All tasks** | 4 |" in markdown
    assert "train** split" in markdown


def test_summary_counts_errors() -> None:
    failed = ExampleResult("x", "QA", True, 1.0, 0, None, 1.0, error="boom")
    assert summarise([failed], 0.5)["scorer_errors"] == 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_end_to_end(ragtruth_dir: Path, tmp_path: Path,
                        capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    code = main(["--data-dir", str(ragtruth_dir), "--per-task", "0", "--dev-per-task", "1",
                 "--out-dir", str(out)])
    assert code == 0
    [run_dir] = list(out.iterdir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["n"] == 15
    assert summary["run"]["threshold_source"].startswith("tuned")
    assert (run_dir / "report.md").exists()
    assert len((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 15
    assert "Groundwire hallucination benchmark" in capsys.readouterr().out


def test_cli_fixed_threshold(ragtruth_dir: Path, tmp_path: Path) -> None:
    code = main(["--data-dir", str(ragtruth_dir), "--per-task", "0", "--dev-per-task", "0",
                 "--threshold", "0.5", "--out-dir", str(tmp_path / "o")])
    assert code == 0


def test_cli_block_budget_and_evidence(ragtruth_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "o"
    code = main(["--data-dir", str(ragtruth_dir), "--per-task", "0", "--dev-per-task", "2",
                 "--max-good-blocked", "0.2", "--evidence-passages", "2",
                 "--aggregation", "min", "--out-dir", str(out)])
    assert code == 0
    [run_dir] = list(out.iterdir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run"]["aggregation"] == "min"
    assert summary["run"]["evidence_passages"] == 2
    assert "block at most 20%" in summary["run"]["threshold_source"]
    first = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(first["claim_scores"], list)


def test_cli_missing_data_returns_1(tmp_path: Path) -> None:
    assert main(["--data-dir", str(tmp_path / "nope"), "--out-dir", str(tmp_path)]) == 1


@pytest.mark.parametrize("argv", [
    ["--threshold", "1.5", "--dev-per-task", "0"],
    ["--threshold", "0.5"],  # conflicts with default tuning
    ["--per-task", "-1"],
    ["--evidence-passages", "11"],
    ["--max-good-blocked", "2"],
    ["--max-good-blocked", "0.2", "--dev-per-task", "0"],
    ["--aggregation", "ratio"],
])
def test_cli_argument_validation(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)
