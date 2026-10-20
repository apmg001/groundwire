"""Tests for benchmark quality gates: spec, checks, runners, reports, CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from groundwire.core.config import GroundingScorerType
from groundwire.core.errors import ConfigurationError, DatasetError
from groundwire.evals.gate import cli
from groundwire.evals.gate.checks import (
    AnyGate,
    CheckResult,
    GateResult,
    evaluate_checks,
    run_gates,
    suggest_bound,
)
from groundwire.evals.gate.report import render_calibration, render_markdown
from groundwire.evals.gate.runners import Metrics, run_qqp_gate, run_ragtruth_gate
from groundwire.evals.gate.spec import Bound, QQPGate, RagTruthGate, load_spec, parse_spec
from groundwire.guardrails.scorers import LexicalScorer
from unit.test_cache_bench import ROWS, KeywordEmbedder, _write_tsv
from unit.test_evals import SOURCES, _responses, _write

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _rag(**overrides: Any) -> RagTruthGate:
    raw: dict[str, Any] = {"name": "rag", "benchmark": "ragtruth", "threshold": 0.5,
                           "per_task": 2, "checks": {"f1": {"min": 0.5}}}
    raw.update(overrides)
    return RagTruthGate.model_validate(raw)


def _qqp(**overrides: Any) -> QQPGate:
    raw: dict[str, Any] = {"name": "qqp", "benchmark": "qqp", "threshold": 0.9, "pairs": 10,
                           "checks": {"false_hit_rate": {"max": 0.1}}}
    raw.update(overrides)
    return QQPGate.model_validate(raw)


def _spec_dict(*gates: dict[str, Any], suite: str = "fast") -> dict[str, Any]:
    return {"suites": {suite: list(gates)}}


RAG_DICT = {"name": "rag", "benchmark": "ragtruth", "threshold": 0.5,
            "checks": {"roc_auc": {"min": 0.7}, "false_block_rate": {"max": 0.4}}}


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


def test_shipped_gate_file_is_valid() -> None:
    spec = load_spec(Path("config/quality_gates.yaml"))
    fast = spec.suite("fast")
    assert fast and not any(g.needs_models for g in fast)  # fast suite stays dependency-free
    assert all(g.needs_models for g in spec.suite("full"))


def test_bound_rules() -> None:
    assert Bound(min=0.5).allows(0.5) and not Bound(min=0.5).allows(0.49)
    assert Bound(max=0.1).allows(0.1) and not Bound(max=0.1).allows(0.11)
    assert Bound(min=0.1, max=0.2).describe() == "0.1 .. 0.2"
    assert Bound(min=0.75).describe() == ">= 0.75" and Bound(max=0).describe() == "<= 0"
    with pytest.raises(ValueError, match="min, max"):
        Bound()
    with pytest.raises(ValueError, match="greater"):
        Bound(min=0.9, max=0.1)


@pytest.mark.parametrize(("gate", "message"), [
    ({**RAG_DICT, "checks": {}}, "checks"),
    ({**RAG_DICT, "checks": {"hit_rate": {"min": 0.1}}}, "checks"),  # QQP metric on RAGTruth
    ({**RAG_DICT, "benchmark": "mmlu"}, "benchmark"),
    ({**RAG_DICT, "threshold": 1.5}, "threshold"),
    ({**RAG_DICT, "name": "Has Spaces"}, "name"),
    ({**RAG_DICT, "surprise": 1}, "surprise"),
])
def test_spec_rejects_bad_gates(gate: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        parse_spec(_spec_dict(gate))


def test_spec_rejects_duplicate_names_and_empty_suites() -> None:
    with pytest.raises(ConfigurationError, match="unique"):
        parse_spec({"suites": {"a": [RAG_DICT], "b": [RAG_DICT]}})
    with pytest.raises(ConfigurationError):
        parse_spec({"suites": {"a": []}})


def test_spec_unknown_suite_and_file_errors(tmp_path: Path) -> None:
    spec = parse_spec(_spec_dict(RAG_DICT))
    with pytest.raises(ConfigurationError, match="unknown suite"):
        spec.suite("nightly")
    with pytest.raises(ConfigurationError, match="not found"):
        load_spec(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("suites: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_spec(bad)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def test_missing_or_nan_metric_fails_closed() -> None:
    gate = _rag(checks={"f1": {"min": 0.5}, "roc_auc": {"min": 0.5}})
    results = evaluate_checks(gate, {"f1": 0.9})
    assert [c.passed for c in results] == [True, False]
    assert not CheckResult("f1", float("nan"), Bound(min=0)).passed


async def test_run_gates_is_fail_closed_and_continues() -> None:
    gates: list[AnyGate] = [_rag(name="ok"), _rag(name="missing-data"), _rag(name="crash"),
                            _rag(name="low")]

    async def runner(gate: AnyGate) -> Metrics:
        if gate.name == "missing-data":
            raise DatasetError("data/ragtruth not found")
        if gate.name == "crash":
            raise RuntimeError("segfault-ish")
        return {"f1": 0.9 if gate.name == "ok" else 0.1}

    results = await run_gates(gates, runner)
    assert [r.passed for r in results] == [True, False, False, False]
    assert results[1].error == "data/ragtruth not found"
    assert results[2].error is not None and "RuntimeError" in results[2].error
    assert results[3].error is None and not results[3].checks[0].passed
    assert results[0].as_dict()["checks"][0] == {"metric": "f1", "value": 0.9, "min": 0.5,
                                                 "max": None, "passed": True}


def test_gate_without_checks_never_passes() -> None:
    assert not GateResult(gate=_rag(), checks=[]).passed


@pytest.mark.parametrize(("metric", "value", "expected"), [
    ("f1", 0.6357, Bound(min=0.615)),
    ("false_block_rate", 0.4124, Bound(max=0.433)),
    ("scorer_errors", 3.0, Bound(max=0)),
    ("roc_auc", 0.01, Bound(min=0.0)),
])
def test_suggest_bound_direction(metric: str, value: float, expected: Bound) -> None:
    assert suggest_bound(_rag(), metric, value, 0.02) == expected


def test_suggest_bound_for_qqp_ceiling_is_capped() -> None:
    assert suggest_bound(_qqp(), "false_hit_rate", 0.995, 0.02) == Bound(max=1.0)
    assert suggest_bound(_qqp(), "hit_rate", 0.149, 0.02) == Bound(min=0.129)


# --------------------------------------------------------------------------- #
# Runners (real benchmark code on tiny fixture data)
# --------------------------------------------------------------------------- #


async def test_ragtruth_runner(tmp_path: Path) -> None:
    root = _write(tmp_path / "ragtruth", SOURCES, _responses())
    gate = _rag(per_task=5, checks={"roc_auc": {"min": 0}})
    metrics = await run_ragtruth_gate(gate, root, scorer=LexicalScorer())
    assert metrics["n"] > 0 and metrics["scorer_errors"] == 0
    assert {"precision", "recall", "f1", "roc_auc", "false_block_rate"} <= set(metrics)


async def test_ragtruth_runner_builds_lexical_scorer(tmp_path: Path) -> None:
    root = _write(tmp_path / "ragtruth", SOURCES, _responses())
    metrics = await run_ragtruth_gate(_rag(scorer=GroundingScorerType.LEXICAL), root)
    assert metrics["n"] > 0


def test_qqp_runner(tmp_path: Path) -> None:
    _write_tsv(tmp_path / "validation.tsv", ROWS)
    metrics = run_qqp_gate(_qqp(pairs=10, threshold=0.99), tmp_path, embedder=KeywordEmbedder())
    assert metrics["n"] == len(ROWS)
    assert 0 <= metrics["false_hit_rate"] <= 1 and 0 <= metrics["roc_auc"] <= 1


def test_qqp_runner_missing_data(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        run_qqp_gate(_qqp(), tmp_path, embedder=KeywordEmbedder())


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def _results() -> list[GateResult]:
    good = GateResult(gate=_rag(name="good"), metrics={"f1": 0.7, "n": 450}, seconds=1.0)
    good.checks = evaluate_checks(good.gate, good.metrics)
    bad = GateResult(gate=_rag(name="bad"), metrics={"f1": 0.2}, seconds=1.0)
    bad.checks = evaluate_checks(bad.gate, bad.metrics)
    broken = GateResult(gate=_qqp(name="broken"), error="data/qqp missing")
    return [good, bad, broken]


def test_markdown_report() -> None:
    text = render_markdown("fast", _results())
    assert "❌ FAILED" in text and "450 examples" in text
    assert "| f1 | 0.2000 | >= 0.5 | **FAIL** |" in text
    assert "Could not run: data/qqp missing" in text
    assert "lower the bound" in text
    assert "✅ PASSED" in render_markdown("fast", _results()[:1])


def test_calibration_output_is_valid_yaml_per_gate() -> None:
    text = render_calibration(_results(), 0.02)
    assert "good:\n  checks:\n    f1:\n      min: 0.68" in text
    assert "# broken: could not run" in text


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _spec_file(tmp_path: Path) -> Path:
    path = tmp_path / "gates.yaml"
    path.write_text(json.dumps(_spec_dict(RAG_DICT)), encoding="utf-8")  # JSON is valid YAML
    return path


def _runner(metrics: Metrics) -> Any:
    async def run(gate: AnyGate) -> Metrics:
        return metrics
    return run


def test_cli_pass_writes_json_and_step_summary(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    code = cli.main(["--spec", str(_spec_file(tmp_path)), "--out-dir", str(tmp_path / "out")],
                    runner=_runner({"roc_auc": 0.8, "false_block_rate": 0.3}))
    assert code == cli.EXIT_OK
    assert "✅ PASSED" in summary.read_text(encoding="utf-8")
    saved = json.loads((tmp_path / "out" / "fast.json").read_text(encoding="utf-8"))
    assert saved[0]["passed"] is True


def test_cli_failure_exit_code(tmp_path: Path) -> None:
    code = cli.main(["--spec", str(_spec_file(tmp_path)), "--out-dir", str(tmp_path / "o")],
                    runner=_runner({"roc_auc": 0.6, "false_block_rate": 0.3}))
    assert code == cli.EXIT_FAILED


def test_cli_usage_errors(tmp_path: Path) -> None:
    assert cli.main(["--spec", str(tmp_path / "nope.yaml")]) == cli.EXIT_USAGE
    assert cli.main(["--spec", str(_spec_file(tmp_path)), "--suite", "x"]) == cli.EXIT_USAGE
    with pytest.raises(SystemExit):
        cli.parse_args(["--tolerance", "0.9"])


def test_cli_calibrate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["--spec", str(_spec_file(tmp_path)), "--calibrate"],
                    runner=_runner({"roc_auc": 0.7736, "false_block_rate": 0.4124}))
    assert code == cli.EXIT_OK  # calibration reports, it does not enforce
    out = capsys.readouterr().out
    assert "min: 0.753" in out and "max: 0.433" in out


def test_cli_calibrate_fails_if_a_gate_cannot_run(tmp_path: Path) -> None:
    async def broken(gate: AnyGate) -> Metrics:
        raise DatasetError("missing")

    assert cli.main(["--spec", str(_spec_file(tmp_path)), "--calibrate"],
                    runner=broken) == cli.EXIT_FAILED
