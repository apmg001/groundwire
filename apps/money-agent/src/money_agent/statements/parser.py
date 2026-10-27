"""Read bank-statement CSV exports into rows.

Real exports are messy, and this parser is built for that:

* **Preamble lines** (bank name, account number, period) before the table:
  the header row is *found*, not assumed to be line 1.
* **Different column names** for the same thing ("Narration", "Particulars",
  "Description"; "Withdrawal Amt.", "Debit", "Dr"): matched through aliases.
* **Two layouts for amounts**: separate debit/credit columns, or one amount
  column with a sign, a ``Dr``/``Cr`` suffix, or a separate type column.
* **Indian formats**: day-first dates (``05/09/2026``, ``05-Sep-26``) and
  lakh grouping (``1,23,456.78``).
* **Footers and notes** after the table are skipped quietly; rows that look
  like transactions but cannot be read are reported (with their line number)
  instead of silently dropped or failing the whole upload.

Limits on size and rows bound memory and CPU per upload. The parser is pure:
bytes in, rows out, no I/O.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum

from money_agent.core.errors import StatementFormatError, StatementTooLargeError
from money_agent.domain.models import Direction
from money_agent.domain.money import InvalidAmountError, Money

logger = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 300
HEADER_SCAN_ROWS = 40
MAX_REPORTED_PROBLEMS = 20

# Tried in order; Indian statements are day-first.
DATE_FORMATS = (
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y", "%Y-%m-%d",
    "%d %b %Y", "%d-%b-%Y", "%d-%b-%y", "%d %b %y", "%d %B %Y", "%b %d, %Y",
)


class Column(StrEnum):
    """Logical columns of a statement."""

    DATE = "date"
    DESCRIPTION = "description"
    DEBIT = "debit"
    CREDIT = "credit"
    AMOUNT = "amount"
    TYPE = "type"
    BALANCE = "balance"


# Normalised header text -> column. Earlier aliases win when a sheet has several
# (a "Txn Date" beats a "Value Date").
ALIASES: dict[Column, tuple[str, ...]] = {
    Column.DATE: ("txn date", "transaction date", "tran date", "date", "posting date",
                  "value date", "txn dt", "value dt"),
    Column.DESCRIPTION: ("description", "narration", "particulars", "transaction details",
                         "details", "transaction remarks", "remarks"),
    Column.DEBIT: ("debit", "withdrawal amount", "withdrawal", "withdrawals",
                   "debit amount", "dr amount", "dr", "paid out", "money out"),
    Column.CREDIT: ("credit", "deposit amount", "deposit", "deposits", "credit amount",
                    "cr amount", "cr", "paid in", "money in"),
    Column.AMOUNT: ("amount", "transaction amount", "txn amount"),
    Column.TYPE: ("type", "dr cr", "cr dr", "transaction type", "debit credit"),
    Column.BALANCE: ("balance", "closing balance", "running balance", "available balance"),
}
_DEBIT_WORDS = frozenset({"dr", "debit", "d", "withdrawal", "out"})
_CREDIT_WORDS = frozenset({"cr", "credit", "c", "deposit", "in"})
_NOISE_WORDS = frozenset({"inr", "rs", "in", "of"})
_SPACES = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ParsedRow:
    """One readable transaction row.

    Attributes:
        line: 1-based line number in the file (for error messages).
        posted_on: Transaction date.
        description: Narration, whitespace-normalised.
        amount: Size of the movement (positive).
        direction: Debit or credit.
        balance: Running balance, if the statement has one.
    """

    line: int
    posted_on: date
    description: str
    amount: Money
    direction: Direction
    balance: Money | None = None


@dataclass(frozen=True, slots=True)
class RowProblem:
    """A row that looked like a transaction but could not be read.

    Attributes:
        line: 1-based line number.
        reason: What was wrong (no row content, which may be personal data).
    """

    line: int
    reason: str


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Everything read from one statement.

    Attributes:
        rows: Readable transactions, in file order.
        problems: Up to ``MAX_REPORTED_PROBLEMS`` unreadable rows.
        problem_count: Total unreadable rows.
        columns: Which file column each logical column came from.
    """

    rows: tuple[ParsedRow, ...]
    problems: tuple[RowProblem, ...] = ()
    problem_count: int = 0
    columns: dict[str, str] = field(default_factory=dict)


class StatementParser:
    """Parses CSV bank statements."""

    def __init__(self, *, max_bytes: int = 5_000_000, max_rows: int = 50_000) -> None:
        """Create a parser.

        Args:
            max_bytes: Largest file accepted.
            max_rows: Most lines accepted.

        Raises:
            ValueError: If a limit is not positive.
        """
        if max_bytes < 1 or max_rows < 1:
            raise ValueError("limits must be positive")
        self._max_bytes = max_bytes
        self._max_rows = max_rows

    def parse(self, data: bytes) -> ParseResult:
        """Parse a statement.

        Args:
            data: The uploaded file's bytes.

        Returns:
            Rows and per-row problems.

        Raises:
            StatementTooLargeError: The file exceeds the size or row limit.
            StatementFormatError: No transaction table found, or no row could be read.
        """
        if len(data) > self._max_bytes:
            raise StatementTooLargeError(f"statement is {len(data)} bytes",
                                         details={"max_bytes": self._max_bytes})
        lines = self._read_lines(data)
        if len(lines) > self._max_rows + HEADER_SCAN_ROWS:
            raise StatementTooLargeError(f"statement has {len(lines)} lines",
                                         details={"max_rows": self._max_rows})
        header_index, mapping = self._find_header(lines)
        rows: list[ParsedRow] = []
        problems: list[RowProblem] = []
        for offset, cells in enumerate(lines[header_index + 1:], start=header_index + 2):
            outcome = _read_row(offset, cells, mapping)
            if isinstance(outcome, ParsedRow):
                rows.append(outcome)
            elif outcome is not None:
                problems.append(outcome)
        if not rows:
            raise StatementFormatError("no readable transaction rows",
                                       details={"unreadable_rows": len(problems)})
        logger.info("statement parsed", extra={"rows": len(rows), "problems": len(problems)})
        header = lines[header_index]
        return ParseResult(rows=tuple(rows), problems=tuple(problems[:MAX_REPORTED_PROBLEMS]),
                           problem_count=len(problems),
                           columns={col.value: header[i].strip() for col, i in mapping.items()})

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_lines(data: bytes) -> list[list[str]]:
        """Decode and split into cells, detecting the delimiter."""
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp1252", errors="replace")
        if not text.strip():
            raise StatementFormatError("statement is empty")
        try:
            dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(
                text[:8_192], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        try:
            return list(csv.reader(io.StringIO(text), dialect))
        except csv.Error as exc:
            raise StatementFormatError(f"malformed CSV: {exc}") from exc

    @staticmethod
    def _find_header(lines: Sequence[list[str]]) -> tuple[int, dict[Column, int]]:
        """The first row (within the scan window) that names the needed columns."""
        for index, cells in enumerate(lines[:HEADER_SCAN_ROWS]):
            mapping = map_columns(cells)
            has_amount = (Column.AMOUNT in mapping or Column.DEBIT in mapping
                          or Column.CREDIT in mapping)
            if Column.DATE in mapping and Column.DESCRIPTION in mapping and has_amount:
                return index, mapping
        raise StatementFormatError("no header row with date, description and amount columns")


def normalise_header(text: str) -> str:
    """``"Withdrawal Amt. (INR)"`` -> ``"withdrawal amount"``.

    Args:
        text: A header cell.

    Returns:
        Lower-case words, with currency and filler words removed.
    """
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    words = ["amount" if w == "amt" else w for w in words if w not in _NOISE_WORDS]
    return " ".join(words)


def map_columns(cells: Sequence[str]) -> dict[Column, int]:
    """Match header cells to logical columns.

    Args:
        cells: One row of the file.

    Returns:
        Logical column -> cell index, for the columns found.
    """
    names = [normalise_header(cell) for cell in cells]
    mapping: dict[Column, int] = {}
    for column, aliases in ALIASES.items():
        for alias in aliases:
            if alias in names and names.index(alias) not in mapping.values():
                mapping[column] = names.index(alias)
                break
    return mapping


def parse_date(text: str) -> date | None:
    """Parse a statement date (day-first), or ``None`` if it is not a date.

    Args:
        text: The cell.

    Returns:
        The date.
    """
    value = _SPACES.sub(" ", text.strip())
    if not value:
        return None
    candidate = value.split(" ")[0] if re.match(r"^\d{1,4}[/.-]", value) else value
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).date()  # a calendar date: no time zone
        except ValueError:
            continue
    return None


def _cell(cells: Sequence[str], mapping: dict[Column, int], column: Column) -> str:
    index = mapping.get(column)
    return cells[index].strip() if index is not None and index < len(cells) else ""


def _amount_or_none(text: str) -> Money | None:
    if not text or text in {"-", "--"}:
        return None
    return Money.parse(text)


def _read_row(line: int, cells: Sequence[str], mapping: dict[Column, int]
              ) -> ParsedRow | RowProblem | None:
    """Read one row.

    Returns:
        The row; a problem if it looks like a transaction but cannot be read;
        ``None`` for blank, footer and note rows.
    """
    if not any(cell.strip() for cell in cells):
        return None
    raw_date = _cell(cells, mapping, Column.DATE)
    try:
        amount, direction = _movement(cells, mapping)
    except InvalidAmountError:
        return RowProblem(line, "unreadable amount") if parse_date(raw_date) else None
    posted_on = parse_date(raw_date)
    if posted_on is None:
        return RowProblem(line, "unreadable date") if amount is not None else None
    if amount is None or direction is None:
        return RowProblem(line, "no amount")
    if not amount:
        return RowProblem(line, "zero amount")
    description = _SPACES.sub(" ", _cell(cells, mapping, Column.DESCRIPTION))
    if not description:
        return RowProblem(line, "no description")
    try:
        balance = _amount_or_none(_cell(cells, mapping, Column.BALANCE))
    except InvalidAmountError:
        balance = None  # a running balance is optional; a bad one is not fatal
    return ParsedRow(line=line, posted_on=posted_on,
                     description=description[:MAX_DESCRIPTION_CHARS],
                     amount=amount, direction=direction, balance=balance)


def _movement(cells: Sequence[str], mapping: dict[Column, int]
              ) -> tuple[Money | None, Direction | None]:
    """Amount and direction, from whichever layout the statement uses.

    Raises:
        InvalidAmountError: An amount cell holds something that is not money.
    """
    if Column.DEBIT in mapping or Column.CREDIT in mapping:
        debit = _amount_or_none(_cell(cells, mapping, Column.DEBIT))
        credit = _amount_or_none(_cell(cells, mapping, Column.CREDIT))
        if debit:
            return abs(debit), Direction.DEBIT
        if credit:
            return abs(credit), Direction.CREDIT
        if Column.AMOUNT not in mapping:
            return (Money.zero(), Direction.DEBIT) if debit is not None or credit is not None \
                else (None, None)
    amount = _amount_or_none(_cell(cells, mapping, Column.AMOUNT))
    if amount is None:
        return None, None
    kind = normalise_header(_cell(cells, mapping, Column.TYPE))
    if kind in _DEBIT_WORDS:
        return abs(amount), Direction.DEBIT
    if kind in _CREDIT_WORDS:
        return abs(amount), Direction.CREDIT
    return abs(amount), Direction.DEBIT if amount.paise < 0 else Direction.CREDIT
