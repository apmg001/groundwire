"""Use case: import a bank statement.

parse (in a worker thread) -> categorise -> fingerprint -> store new rows.

* **Never double-counts.** Each row gets a fingerprint from its date,
  direction, amount and narration, plus its position among identical rows in
  the same file (two ₹20 teas at the same stall on the same day are two
  transactions). Re-uploading a file, or uploading overlapping periods,
  inserts only the rows not already stored.
* **Never blocks the server.** Parsing a 50,000-row file is CPU work, so it
  runs in a thread; other users' requests keep being served meanwhile.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from collections import Counter
from dataclasses import dataclass

from money_agent.domain.models import Transaction
from money_agent.domain.ports import TransactionRepository, UserRepository
from money_agent.statements.categoriser import Categoriser
from money_agent.statements.parser import ParsedRow, ParseResult, RowProblem, StatementParser

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImportReport:
    """What an upload did.

    Attributes:
        rows_read: Readable rows in the file.
        imported: New transactions stored.
        duplicates: Rows already stored by an earlier upload.
        problems: Unreadable rows (line numbers and reasons, first few).
        problem_count: All unreadable rows.
        first_date: Earliest date in the file.
        last_date: Latest date in the file.
        columns: Which file columns were used.
    """

    rows_read: int
    imported: int
    duplicates: int
    problems: tuple[RowProblem, ...]
    problem_count: int
    first_date: str
    last_date: str
    columns: dict[str, str]


def fingerprint(row: ParsedRow, ordinal: int) -> str:
    """Identity of a statement row across uploads.

    Args:
        row: The row.
        ordinal: How many identical rows came before it in the same file.

    Returns:
        A hex SHA-256 digest.
    """
    narration = " ".join(row.description.lower().split())
    key = (f"{row.posted_on.isoformat()}|{row.direction.value}|{row.amount.paise}|"
           f"{narration}|{ordinal}")
    return hashlib.sha256(key.encode()).hexdigest()


class StatementService:
    """Imports statements for a user."""

    def __init__(self, users: UserRepository, transactions: TransactionRepository,
                 parser: StatementParser, categoriser: Categoriser) -> None:
        """Create the service.

        Args:
            users: User storage.
            transactions: Transaction storage.
            parser: Statement parser.
            categoriser: Category rules.
        """
        self._users = users
        self._transactions = transactions
        self._parser = parser
        self._categoriser = categoriser

    async def import_statement(self, user_id: str, data: bytes) -> ImportReport:
        """Parse and store a statement.

        Args:
            user_id: Owner.
            data: The uploaded file.

        Returns:
            The import report.

        Raises:
            StatementTooLargeError: Over the size or row limit.
            StatementFormatError: Not a readable statement.
            StorageUnavailableError: The store failed.
        """
        parsed, transactions = await asyncio.to_thread(self._prepare, user_id, data)
        await self._users.ensure(user_id)
        imported = await self._transactions.add_many(user_id, transactions)
        dates = [row.posted_on for row in parsed.rows]
        report = ImportReport(
            rows_read=len(parsed.rows), imported=imported,
            duplicates=len(parsed.rows) - imported, problems=parsed.problems,
            problem_count=parsed.problem_count, first_date=min(dates).isoformat(),
            last_date=max(dates).isoformat(), columns=parsed.columns)
        logger.info("statement imported", extra={
            "rows": report.rows_read, "imported": imported, "duplicates": report.duplicates,
            "problems": report.problem_count})
        return report

    def _prepare(self, user_id: str, data: bytes) -> tuple[ParseResult, list[Transaction]]:
        """CPU part of the import: parse, categorise, fingerprint (runs in a thread)."""
        parsed = self._parser.parse(data)
        seen: Counter[tuple[object, ...]] = Counter()
        transactions: list[Transaction] = []
        for row in parsed.rows:
            identity = (row.posted_on, row.direction, row.amount.paise,
                        " ".join(row.description.lower().split()))
            transactions.append(Transaction(
                id=uuid.uuid4(), user_id=user_id, posted_on=row.posted_on,
                description=row.description, amount=row.amount, direction=row.direction,
                category=self._categoriser.categorise(row.description, row.direction),
                fingerprint=fingerprint(row, seen[identity]), balance=row.balance))
            seen[identity] += 1
        return parsed, transactions
