"""Benchmark dataset loading.

Currently supports **RAGTruth** (Niu et al., 2024; MIT licence): ~18K LLM
responses to retrieval-augmented prompts, human-annotated for hallucinated
spans, across three tasks:

* ``QA``: answer a question from retrieved web passages,
* ``Summary``: summarise a news article,
* ``Data2txt``: write a description from structured business data (JSON).

This matches Groundwire's use case exactly: a model answers from supplied
context, and the question is whether the answer stays within it.

A response is labelled *hallucinated* when it contains at least one annotated
span. :class:`LabelPolicy` controls how two annotator flags are treated:

* ``implicit_true``: the span is factually true but not stated in the context.
* ``due_to_null``: the span comes from the model misreading a null field.

The default ``strict`` policy counts every annotated span, as the RAGTruth
paper does. ``exclude_implicit_true`` ignores spans that are true but
unsupported, which is closer to "is this answer wrong?" than "is it grounded?".

Expected layout (a clone of github.com/ParticleMedia/RAGTruth)::

    <root>/dataset/response.jsonl
    <root>/dataset/source_info.jsonl
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from groundwire.core.errors import DatasetError

logger = logging.getLogger(__name__)

RAGTRUTH_TASKS: tuple[str, ...] = ("QA", "Summary", "Data2txt")


class LabelPolicy(StrEnum):
    """How annotated spans map to a response-level label."""

    STRICT = "strict"
    EXCLUDE_IMPLICIT_TRUE = "exclude_implicit_true"


@dataclass(frozen=True, slots=True)
class EvalExample:
    """One benchmark example.

    Attributes:
        id: Dataset-specific identifier.
        task: Task name (e.g. ``"QA"``).
        context: Source passages the response should be grounded in.
        response: Model response under test.
        is_hallucinated: Ground-truth label.
        meta: Extra fields kept for analysis (generating model, span types).
        prompt: The exact prompt the response was generated from, if known
            (used verbatim by token detectors).
    """

    id: str
    task: str
    context: tuple[str, ...]
    response: str
    is_hallucinated: bool
    meta: dict[str, Any] = field(default_factory=dict)
    prompt: str | None = None


# --------------------------------------------------------------------------- #
# Context formatting
# --------------------------------------------------------------------------- #


def _flatten(value: Any, prefix: str = "") -> Iterator[str]:
    """Turn nested JSON into ``"key: value."`` lines a scorer can read.

    Args:
        value: JSON value.
        prefix: Dotted key path so far.

    Yields:
        One readable line per leaf value.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value, start=1):
            yield from _flatten(child, f"{prefix} {index}" if prefix else str(index))
    elif value is None:
        yield f"{prefix}: not provided."
    else:
        text = str(value).strip()
        yield f"{prefix}: {text}" + ("" if text.endswith((".", "!", "?")) else ".")


def ragtruth_context(task: str, source_info: Any) -> tuple[str, ...]:
    """Build context passages for a RAGTruth source record.

    Args:
        task: ``"QA"``, ``"Summary"`` or ``"Data2txt"``.
        source_info: The record's ``source_info`` field.

    Returns:
        Context passages.

    Raises:
        DatasetError: If the record shape does not match the task.
    """
    if task == "QA":
        if not isinstance(source_info, dict) or "passages" not in source_info:
            raise DatasetError("QA source_info must be an object with 'passages'")
        # The question is not evidence; only the retrieved passages are.
        passages = [p.strip() for p in str(source_info["passages"]).split("\n\n") if p.strip()]
        return tuple(passages)
    if task == "Summary":
        if not isinstance(source_info, str):
            raise DatasetError("Summary source_info must be a string")
        return (source_info,)
    if task == "Data2txt":
        if not isinstance(source_info, dict):
            raise DatasetError("Data2txt source_info must be an object")
        lines = list(_flatten(source_info))
        return tuple(lines) if lines else ("",)
    raise DatasetError(f"unknown RAGTruth task '{task}'")


def ragtruth_label(labels: Sequence[dict[str, Any]], policy: LabelPolicy) -> bool:
    """Compute the response-level label from annotated spans.

    Args:
        labels: The response's ``labels`` list.
        policy: Label policy.

    Returns:
        True if the response counts as hallucinated.
    """
    if policy is LabelPolicy.EXCLUDE_IMPLICIT_TRUE:
        return any(not span.get("implicit_true", False) for span in labels)
    return bool(labels)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file.

    Args:
        path: File path.

    Yields:
        Parsed objects.

    Raises:
        DatasetError: If the file is missing or a line is not a JSON object.
    """
    try:
        handle = path.open(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DatasetError(
            f"{path} not found. Download the dataset first: make bench-data"
        ) from exc
    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise DatasetError(f"{path}:{line_no}: expected a JSON object")
            yield record


def load_ragtruth(
    root: Path,
    *,
    split: str = "test",
    tasks: Iterable[str] = RAGTRUTH_TASKS,
    per_task: int | None = None,
    seed: int = 13,
    label_policy: LabelPolicy = LabelPolicy.STRICT,
) -> list[EvalExample]:
    """Load RAGTruth examples, optionally sampling a fixed number per task.

    Sampling is seeded, so the same arguments always give the same examples
    (reproducible benchmark numbers).

    Args:
        root: RAGTruth clone directory.
        split: ``"train"`` or ``"test"``.
        tasks: Tasks to include.
        per_task: Examples per task (random sample); ``None`` loads all.
        seed: Sampling seed.
        label_policy: How annotated spans become a label.

    Returns:
        Examples ordered by task, then by sample order.

    Raises:
        DatasetError: If files are missing/malformed, or a task/split is invalid.
    """
    wanted = tuple(tasks)
    unknown = [t for t in wanted if t not in RAGTRUTH_TASKS]
    if unknown:
        raise DatasetError(f"unknown tasks {unknown}; choose from {list(RAGTRUTH_TASKS)}")
    if split not in ("train", "test"):
        raise DatasetError(f"split must be 'train' or 'test', got '{split}'")
    if per_task is not None and per_task < 1:
        raise DatasetError("per_task must be >= 1 (or None for all)")

    dataset_dir = root / "dataset"
    sources: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(dataset_dir / "source_info.jsonl"):
        try:
            sources[str(record["source_id"])] = record
        except KeyError as exc:
            raise DatasetError("source_info record without 'source_id'") from exc

    by_task: dict[str, list[EvalExample]] = defaultdict(list)
    skipped = 0
    for record in _read_jsonl(dataset_dir / "response.jsonl"):
        if record.get("split") != split:
            continue
        source = sources.get(str(record.get("source_id")))
        if source is None:
            skipped += 1
            continue
        task = str(source.get("task_type"))
        if task not in wanted:
            continue
        labels = record.get("labels") or []
        try:
            context = ragtruth_context(task, source.get("source_info"))
        except DatasetError:
            skipped += 1
            continue
        by_task[task].append(EvalExample(
            id=str(record.get("id")),
            task=task,
            context=context,
            response=str(record.get("response", "")),
            is_hallucinated=ragtruth_label(labels, label_policy),
            meta={"model": record.get("model"), "source_id": str(record.get("source_id")),
                  "span_types": sorted({str(s.get("label_type")) for s in labels})},
            prompt=str(source["prompt"]) if source.get("prompt") else None,
        ))

    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not crypto
    examples: list[EvalExample] = []
    for task in wanted:
        pool = by_task.get(task, [])
        if not pool:
            raise DatasetError(f"no '{split}' examples found for task '{task}'")
        chosen = rng.sample(pool, per_task) if per_task and per_task < len(pool) else pool
        examples.extend(chosen)

    logger.info("RAGTruth loaded", extra={
        "split": split, "examples": len(examples), "skipped": skipped,
        "hallucinated": sum(e.is_hallucinated for e in examples),
        "label_policy": label_policy.value,
    })
    return examples
