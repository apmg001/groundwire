"""Tests for the cascade simulation: pairing, policies, cost model, report, CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from groundwire.evals.cascade import cli
from groundwire.evals.cascade.report import build_summary, render_markdown
from groundwire.evals.cascade.simulate import (
    CascadeCase,
    PolicyOutcome,
    build_cases,
    cascade,
    cheap_only,
    detector_quality,
    oracle,
    pair_by_source,
    strong_only,
)
from groundwire.evals.datasets import EvalExample
from groundwire.evals.runner import ExampleResult
from groundwire.guardrails.scorers import LexicalScorer

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _example(source: str, model: str, hallucinated: bool, task: str = "QA") -> EvalExample:
    return EvalExample(id=f"{source}-{model}", task=task, context=("ctx",), response="r",
                       is_hallucinated=hallucinated,
                       meta={"model": model, "source_id": source})


def _case(score: float, cheap_bad: bool, strong_bad: bool = False, *, error: bool = False,
          task: str = "QA") -> CascadeCase:
    return CascadeCase(source_id="s", task=task, cheap_score=score, cheap_error=error,
                       cheap_hallucinated=cheap_bad, strong_hallucinated=strong_bad)


# Ten prompts: 4 bad cheap answers (3 detected), 6 good (1 wrongly flagged); strong
# model wrong once, on a prompt where the cheap answer was fine.
CASES = [
    _case(0.1, True), _case(0.2, True), _case(0.1, True), _case(0.9, True),
    _case(0.2, False), _case(0.9, False), _case(0.9, False, strong_bad=True),
    _case(0.8, False), _case(0.7, False), _case(0.95, False),
]


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #


def test_pair_by_source_matches_models_per_prompt() -> None:
    examples = [_example("1", "cheap", True), _example("1", "strong", False),
                _example("1", "other", True), _example("2", "cheap", False),
                _example("3", "strong", False)]  # 2 and 3 lack a partner
    pairs = pair_by_source(examples, "cheap", "strong")
    assert [(c.id, s.id) for c, s in pairs] == [("1-cheap", "1-strong")]


def test_pair_by_source_errors() -> None:
    with pytest.raises(ValueError, match="differ"):
        pair_by_source([], "a", "a")
    with pytest.raises(ValueError, match="no prompt"):
        pair_by_source([_example("1", "cheap", True)], "cheap", "strong")


def test_build_cases_checks_alignment() -> None:
    pairs = [(_example("1", "cheap", True), _example("1", "strong", False))]
    good = ExampleResult("1-cheap", "QA", True, 0.2, 2, None, 1.0)
    case = build_cases(pairs, [good])[0]
    assert (case.cheap_score, case.cheap_hallucinated, case.strong_hallucinated) == \
        (0.2, True, False)
    with pytest.raises(ValueError, match="does not match"):
        build_cases(pairs, [ExampleResult("x", "QA", True, 0.2, 2, None, 1.0)])
    with pytest.raises(ValueError, match="pairs but"):
        build_cases(pairs, [])


# --------------------------------------------------------------------------- #
# Policies and cost
# --------------------------------------------------------------------------- #


def test_policies_on_a_known_set() -> None:
    assert cheap_only(CASES).hallucinated == 4
    assert strong_only(CASES).hallucinated == 1
    casc = cascade(CASES, 0.5)
    assert casc.escalated == 4  # three bad caught + one good wasted
    assert casc.hallucinated == 1  # the missed bad answer at 0.9
    best = oracle(CASES)
    assert best.escalated == 4 and best.hallucinated == 0  # beats strong-only


def test_relative_cost() -> None:
    casc = cascade(CASES, 0.5)
    # 10 cheap calls + 4 strong calls at R=10, vs 10 strong calls -> (10 + 40) / 100
    assert casc.relative_cost(10) == pytest.approx(0.5)
    assert strong_only(CASES).relative_cost(10) == pytest.approx(1.0)
    assert cheap_only(CASES).relative_cost(10) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        casc.relative_cost(0)
    assert PolicyOutcome("x", 0, 0, 0, 0, 0).relative_cost(5) == 0.0


def test_unscorable_answers_escalate() -> None:
    assert _case(1.0, False, error=True).escalates(0.5)
    assert not _case(1.0, False).escalates(0.5)


def test_detector_quality() -> None:
    q = detector_quality(CASES, 0.5)
    assert (q.caught, q.missed, q.wasted, q.kept) == (3, 1, 1, 5)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def test_summary_and_markdown() -> None:
    cases = [*CASES, _case(0.1, True, task="Summary")]
    summary = build_summary(cases, threshold=0.5, ratios=[5.0, 15.0], sweep=[0.3, 0.5],
                            run_info={"cheap": "c", "strong": "s", "scorer": "lexical",
                                      "date": "d"})
    names = [p["policy"] for p in summary["policies"]]
    assert names == ["cheap_only", "cascade", "oracle", "strong_only"]
    assert set(summary["by_task"]) == {"QA", "Summary"}
    assert [s["threshold"] for s in summary["sweep"]] == [0.3, 0.5]
    json.dumps(summary)
    text = render_markdown(summary)
    assert "**Result (R=15):**" in text and "Perfect detector (ceiling)" in text
    assert "| QA | **Cascade (Groundwire)** |" in text
    with pytest.raises(ValueError):
        build_summary([], threshold=0.5, ratios=[5.0], sweep=[], run_info={})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _ragtruth(root: Path) -> Path:
    """Two prompts per task, answered by both default models."""
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    sources: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    passages = "The refund for order 7781 is 49.99 USD.\n\nIt was issued on 2 May."
    infos = {"QA": {"question": "q", "passages": passages}, "Summary": passages,
             "Data2txt": {"name": "Cafe", "rating": 4}}
    for task, info in infos.items():
        for n in range(2):
            sid = f"{task}{n}"
            sources.append({"source_id": sid, "task_type": task, "source_info": info,
                            "prompt": "p"})
            for model, bad in (("llama-2-7b-chat", n == 0), ("gpt-4-0613", False)):
                text = "The refund is 89.99 USD." if bad else "The refund is 49.99 USD."
                responses.append({"id": f"{sid}-{model}", "source_id": sid, "model": model,
                                  "split": "test", "response": text,
                                  "labels": [{"label_type": "x"}] if bad else []})
    (dataset / "source_info.jsonl").write_text(
        "\n".join(json.dumps(s) for s in sources), encoding="utf-8")
    (dataset / "response.jsonl").write_text(
        "\n".join(json.dumps(r) for r in responses), encoding="utf-8")
    return root


def test_cli_end_to_end(tmp_path: Path) -> None:
    root = _ragtruth(tmp_path / "rt")
    args = cli.parse_args(["--data-dir", str(root), "--out-dir", str(tmp_path / "out"),
                           "--price-ratios", "10"])
    out_dir = cli.run(args, scorer=LexicalScorer())
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["policies"][0]["prompts"] == 6
    assert summary["threshold"] == pytest.approx(0.3014)
    assert len((out_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_sample_pairs_per_task(tmp_path: Path) -> None:
    root = _ragtruth(tmp_path / "rt")
    args = cli.parse_args(["--data-dir", str(root), "--out-dir", str(tmp_path / "out"),
                           "--per-task", "1"])
    out_dir = cli.run(args, scorer=LexicalScorer())
    assert len((out_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_cli_missing_data(tmp_path: Path) -> None:
    assert cli.main(["--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")]) == 1


@pytest.mark.parametrize("argv", [["--cheap", "gpt-4-0613"], ["--threshold", "2"],
                                  ["--per-task", "-1"], ["--price-ratios", "0"],
                                  ["--price-ratios", "abc"]])
def test_cli_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(argv)
