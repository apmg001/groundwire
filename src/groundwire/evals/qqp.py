"""Quora Question Pairs (QQP) loading for the semantic-cache benchmark.

QQP contains ~400K pairs of real Quora questions, each labelled by humans as
*duplicate* (same intent: one answer serves both) or *not duplicate*. Many
non-duplicates are deliberately close in topic ("How do I learn Python?" vs
"How do I learn Java?"), which is exactly the case a semantic cache must not
confuse. That makes QQP a good stress test for "would reusing the cached
answer have been correct?"

We use the GLUE release hosted on Hugging Face (``nyu-mll/glue``, config
``qqp``): ``train`` (~364K labelled pairs) is used only to *tune* the
threshold, and ``validation`` (~40K labelled pairs) is used to *report*.
GLUE's ``test`` split has no public labels and is not used.

The dataset is downloaded by you (``make bench-cache-data``) and never
committed to this repository; its use is subject to Quora's terms.

Supported file formats:

* ``.parquet`` (the GLUE download; needs ``pyarrow``: ``pip install -e ".[bench]"``)
  with columns ``question1``, ``question2``, ``label``, ``idx``;
* ``.tsv`` with a header containing ``question1``, ``question2`` and either
  ``label`` or ``is_duplicate`` (the original Quora layout; no extra deps).
"""

from __future__ import annotations

import csv
import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groundwire.core.errors import DatasetError

logger = logging.getLogger(__name__)

QQP_SPLITS: tuple[str, ...] = ("train", "validation")


@dataclass(frozen=True, slots=True)
class QuestionPair:
    """One labelled question pair.

    Attributes:
        id: Dataset identifier.
        question1: Question assumed to be in the cache.
        question2: Incoming question looked up against it.
        is_duplicate: Ground truth: one answer serves both.
    """

    id: str
    question1: str
    question2: str
    is_duplicate: bool


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #


def _parse_label(value: Any, where: str) -> bool:
    """Parse a 0/1 label.

    Args:
        value: Raw label value.
        where: Location for error messages.

    Returns:
        The label.

    Raises:
        DatasetError: If the label is not 0 or 1 (e.g. GLUE's unlabeled ``-1``).
    """
    text = str(value).strip()
    if text not in ("0", "1"):
        raise DatasetError(f"{where}: label must be 0 or 1, got {text!r}")
    return text == "1"


def _read_tsv(path: Path) -> Iterator[dict[str, Any]]:
    """Yield rows of a TSV file with a header.

    Args:
        path: TSV path.

    Yields:
        Row dicts.

    Raises:
        DatasetError: If the file is missing or lacks required columns.
    """
    try:
        handle = path.open(encoding="utf-8", newline="")
    except FileNotFoundError as exc:
        raise DatasetError(f"{path} not found. Download it first: make bench-cache-data") from exc
    with handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        columns = set(reader.fieldnames or [])
        if not {"question1", "question2"} <= columns or not columns & {"label", "is_duplicate"}:
            raise DatasetError(
                f"{path}: expected columns question1, question2 and label/is_duplicate, "
                f"got {sorted(columns)}")
        for row in reader:
            row.setdefault("label", row.get("is_duplicate"))
            yield row


def _read_parquet(path: Path) -> Iterator[dict[str, Any]]:
    """Yield rows of a GLUE QQP parquet file.

    Args:
        path: Parquet path.

    Yields:
        Row dicts.

    Raises:
        DatasetError: If pyarrow is missing, or the file is missing or unreadable.
    """
    # Check the file first: "download the data" is the more useful message when
    # both the data and pyarrow are missing (e.g. a fresh clone or CI).
    if not path.is_file():
        raise DatasetError(f"{path} not found. Download it first: make bench-cache-data")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetError('reading .parquet needs pyarrow: pip install -e ".[bench]"') from exc
    try:
        table = pq.read_table(path, columns=["question1", "question2", "label", "idx"])
    except (OSError, ValueError, KeyError) as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = table.to_pylist()
    yield from rows


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    """Dispatch on file extension.

    Args:
        path: Data file.

    Yields:
        Row dicts.

    Raises:
        DatasetError: On an unsupported extension or unreadable file.
    """
    if path.suffix == ".parquet":
        yield from _read_parquet(path)
    elif path.suffix == ".tsv":
        yield from _read_tsv(path)
    else:
        raise DatasetError(f"unsupported QQP file type '{path.suffix}' (use .parquet or .tsv)")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def qqp_path(data_dir: Path, split: str) -> Path:
    """Locate a split file, preferring parquet over TSV.

    Args:
        data_dir: Directory holding the downloaded files.
        split: ``"train"`` or ``"validation"``.

    Returns:
        Path to the split file (may not exist; the loader reports that).

    Raises:
        DatasetError: If the split name is invalid.
    """
    if split not in QQP_SPLITS:
        raise DatasetError(f"split must be one of {list(QQP_SPLITS)}, got '{split}'")
    parquet = data_dir / f"{split}.parquet"
    tsv = data_dir / f"{split}.tsv"
    return parquet if parquet.exists() or not tsv.exists() else tsv


def load_qqp(path: Path, *, limit: int | None = None, seed: int = 13) -> list[QuestionPair]:
    """Load labelled question pairs, optionally a seeded random sample.

    Rows with an empty question are skipped (QQP has a handful).

    Args:
        path: ``.parquet`` or ``.tsv`` file.
        limit: Number of pairs to sample; ``None`` loads all.
        seed: Sampling seed (same arguments always give the same pairs).

    Returns:
        The pairs.

    Raises:
        DatasetError: If the file is missing/malformed, has no usable rows, or
            ``limit`` is invalid.
    """
    if limit is not None and limit < 1:
        raise DatasetError("limit must be >= 1 (or None for all)")
    pairs: list[QuestionPair] = []
    skipped = 0
    for line_no, row in enumerate(_rows(path), start=2):
        q1 = str(row.get("question1") or "").strip()
        q2 = str(row.get("question2") or "").strip()
        if not q1 or not q2:
            skipped += 1
            continue
        label = _parse_label(row.get("label"), f"{path.name}:{line_no}")
        pairs.append(QuestionPair(id=str(row.get("idx", row.get("id", line_no - 2))),
                                  question1=q1, question2=q2, is_duplicate=label))
    if not pairs:
        raise DatasetError(f"{path}: no usable question pairs")
    if limit is not None and limit < len(pairs):
        pairs = random.Random(seed).sample(pairs, limit)  # noqa: S311 - reproducible sampling
    logger.info("QQP loaded", extra={"file": path.name, "pairs": len(pairs), "skipped": skipped,
                                     "duplicates": sum(p.is_duplicate for p in pairs)})
    return pairs
