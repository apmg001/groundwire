"""HTTP routes. Thin by design: identify the caller, call a service, shape the answer.

All user routes live under ``/v1/me``: the user is whoever the identity
provider says sent the request, so there is no user id in any URL to tamper
with. Business rules live in the services; this module only translates HTTP.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse

from money_agent.api.schemas import (
    ChatOut,
    ChatRequest,
    ImportOut,
    MemoryOut,
    MessageOut,
    MonthOut,
    OverviewOut,
    SummaryOut,
    TransactionOut,
    TransactionPageOut,
)
from money_agent.container import Container
from money_agent.core.errors import (
    InvalidInputError,
    StatementTooLargeError,
    StorageUnavailableError,
)
from money_agent.domain.models import (
    MAX_PAGE_SIZE,
    Category,
    Cursor,
    Direction,
    MemoryKind,
    TransactionQuery,
)

UPLOAD_CHUNK = 64 * 1024

router = APIRouter()


def get_container(request: Request) -> Container:
    """The application's container (set by the app factory)."""
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


async def current_user(request: Request, container: ContainerDep) -> str:
    """The calling user's id, from the identity provider.

    Raises:
        AuthenticationError: The caller could not be identified.
    """
    return await container.identity.identify(request.headers)


UserDep = Annotated[str, Depends(current_user)]


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


@router.get("/healthz", tags=["health"])
async def healthz() -> dict[str, str]:
    """Liveness: the process is up (no dependency checks, so restarts aren't cascaded)."""
    return {"status": "ok"}


@router.get("/readyz", tags=["health"])
async def readyz(container: ContainerDep) -> JSONResponse:
    """Readiness: storage answers. A failing replica is taken out of rotation."""
    try:
        await container.health.ping()
    except StorageUnavailableError:  # the cause is logged by the adapter
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return JSONResponse({"status": "ready"})


# --------------------------------------------------------------------------- #
# Statements and transactions
# --------------------------------------------------------------------------- #


@router.post("/v1/me/statements", status_code=status.HTTP_201_CREATED, tags=["statements"])
async def upload_statement(user: UserDep, container: ContainerDep,
                           file: Annotated[UploadFile, File()]) -> ImportOut:
    """Upload a bank statement (CSV); re-uploading never double-counts."""
    limit = container.settings.max_statement_bytes
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(UPLOAD_CHUNK):  # stop reading as soon as it is too big
        size += len(chunk)
        if size > limit:
            raise StatementTooLargeError(f"upload over {limit} bytes",
                                         details={"max_bytes": limit})
        chunks.append(chunk)
    report = await container.statements.import_statement(user, b"".join(chunks))
    return ImportOut.of(report)


@router.get("/v1/me/transactions", tags=["statements"])
async def list_transactions(
        user: UserDep, container: ContainerDep,
        category: Category | None = None, direction: Direction | None = None,
        search: Annotated[str | None, Query(max_length=60)] = None,
        start: date | None = None, end: date | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
        cursor: Annotated[str | None, Query(max_length=200)] = None) -> TransactionPageOut:
    """Transactions, newest first, with keyset pagination (pass ``next_cursor`` back)."""
    try:
        after = Cursor.decode(cursor) if cursor else None
        query = TransactionQuery(start=start, end=end, category=category, direction=direction,
                                 search=search, limit=limit, after=after)
    except ValueError as exc:
        raise InvalidInputError(str(exc), public_message="Invalid filter or cursor.") from exc
    page = await container.finance.transactions(user, query)
    return TransactionPageOut(items=[TransactionOut.of(t) for t in page.items],
                              next_cursor=page.next_cursor.encode() if page.next_cursor
                              else None)


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #


@router.get("/v1/me/overview", tags=["insights"])
async def overview(user: UserDep, container: ContainerDep) -> OverviewOut:
    """Whether the user has uploaded data, and which dates it covers."""
    data = await container.finance.overview(user)
    return OverviewOut(has_data=data is not None,
                       first_date=data.first.isoformat() if data else None,
                       last_date=data.last.isoformat() if data else None)


@router.get("/v1/me/summary", tags=["insights"])
async def summary(user: UserDep, container: ContainerDep,
                  month: Annotated[str | None, Query(pattern=r"^\d{4}-\d{2}$")] = None,
                  start: date | None = None, end: date | None = None) -> SummaryOut:
    """Income, spending and categories (latest month with data by default)."""
    period = await container.finance.resolve_period(user, month=month, start=start, end=end)
    return SummaryOut.of(await container.finance.summary(user, period))


@router.get("/v1/me/trend", tags=["insights"])
async def trend(user: UserDep, container: ContainerDep,
                months: Annotated[int, Query(ge=1, le=24)] = 6) -> list[MonthOut]:
    """Month-by-month figures, oldest first."""
    return [MonthOut.of(m) for m in await container.finance.monthly_trend(user, months)]


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


@router.get("/v1/me/memories", tags=["memory"])
async def memories(user: UserDep, container: ContainerDep,
                   kind: MemoryKind | None = None) -> list[MemoryOut]:
    """Everything the agent remembers about the user."""
    return [MemoryOut.of(m) for m in await container.memory.all(user, kind)]


@router.delete("/v1/me/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT,
               tags=["memory"])
async def forget(user: UserDep, container: ContainerDep, memory_id: uuid.UUID) -> None:
    """Delete a memory."""
    await container.memory.forget(user, memory_id)


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


@router.post("/v1/me/chat", tags=["chat"])
async def chat(user: UserDep, container: ContainerDep, body: ChatRequest) -> ChatOut:
    """Send a message; the reply says whether its figures were verified."""
    reply = await container.chat.send(user, body.message, body.conversation_id)
    return ChatOut.of(reply)


@router.get("/v1/me/conversations/{conversation_id}/messages", tags=["chat"])
async def conversation(user: UserDep, container: ContainerDep, conversation_id: uuid.UUID,
                       limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[MessageOut]:
    """A conversation's recent messages, oldest first."""
    return [MessageOut.of(m) for m in await container.chat.history(user, conversation_id,
                                                                    limit)]
