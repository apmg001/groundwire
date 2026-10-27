"""Long-term memory: what the agent knows about a user across conversations.

Three operations, each a policy decision as much as a storage call:

* **remember**: a new memory that is almost the same as an existing one of
  the same kind *replaces* it ("saving for a bike by March" -> "... by April"),
  so goals update instead of piling up contradictions. Each user has a hard
  cap, so one account cannot grow the database without bound.
* **recall**: what goes into the prompt for a new message. Goals and
  preferences are always included (they shape every answer and are few);
  facts are included when they are relevant to the message. The total is
  capped, so the prompt stays small however long the user has been around.
* **forget**: users can delete anything, through the agent or the API.

Memories are short sentences written by the agent's ``save_memory`` tool,
never raw chat logs, so the store holds what matters and nothing else.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from dataclasses import dataclass, replace

from money_agent.core.errors import InvalidInputError, LimitReachedError, NotFoundError
from money_agent.domain.models import MAX_MEMORY_CHARS, Memory, MemoryKind
from money_agent.domain.ports import Clock, Embedder, MemoryRepository

logger = logging.getLogger(__name__)

_SPACES = re.compile(r"\s+")
ALWAYS_RECALLED = (MemoryKind.GOAL, MemoryKind.PREFERENCE)
MAX_ALWAYS_RECALLED = 8


@dataclass(frozen=True, slots=True)
class RememberResult:
    """Outcome of :meth:`MemoryService.remember`.

    Attributes:
        memory: The stored memory.
        created: True if new; False if it replaced a near-duplicate.
        replaced: The old content, when it replaced one.
    """

    memory: Memory
    created: bool
    replaced: str | None = None


class MemoryService:
    """Remember, recall and forget per-user memories."""

    def __init__(self, repository: MemoryRepository, embedder: Embedder, clock: Clock, *,
                 max_per_user: int = 500, duplicate_similarity: float = 0.8,
                 min_relevance: float = 0.15) -> None:
        """Create the service.

        Args:
            repository: Memory storage.
            embedder: Text-to-vector model.
            clock: Time source.
            max_per_user: Most memories one user may have.
            duplicate_similarity: At or above this similarity (same kind), a new
                memory replaces the existing one.
            min_relevance: Facts below this similarity to a message are not recalled.

        Raises:
            ValueError: If a threshold is outside 0..1 or the cap is not positive.
        """
        if not (0 < duplicate_similarity <= 1 and 0 <= min_relevance < 1) or max_per_user < 1:
            raise ValueError("thresholds must be within 0..1 and the cap positive")
        self._repo = repository
        self._embedder = embedder
        self._clock = clock
        self._max = max_per_user
        self._duplicate = duplicate_similarity
        self._relevance = min_relevance

    async def remember(self, user_id: str, kind: MemoryKind, content: str, *,
                       replaces: uuid.UUID | None = None) -> RememberResult:
        """Store a memory, replacing an older version of it.

        The older version is ``replaces`` when the caller knows it (the agent
        sees memory ids, so "move my bike goal to June" names the goal), else a
        near-duplicate of the same kind, if there is one.

        Args:
            user_id: Owner.
            kind: Goal, fact or preference.
            content: One short sentence.
            replaces: Id of the memory this one supersedes.

        Returns:
            What was stored.

        Raises:
            InvalidInputError: Empty or too long.
            NotFoundError: ``replaces`` is not one of the user's memories.
            LimitReachedError: The user has reached the memory cap.
            StorageUnavailableError: The store failed.
        """
        text = _SPACES.sub(" ", content).strip()
        if not text or len(text) > MAX_MEMORY_CHARS:
            raise InvalidInputError(f"memory length {len(text)}",
                                    public_message=f"A memory must be 1 to {MAX_MEMORY_CHARS} "
                                                   "characters.")
        [vector] = await self._embedder.embed([text])
        now = self._clock.now()
        old = await self._previous_version(user_id, kind, vector, replaces)
        if old is not None:
            updated = replace(old, kind=kind, content=text, updated_at=now)
            await self._repo.replace(updated, vector)
            logger.info("memory replaced", extra={"kind": kind.value, "memory_id": str(old.id)})
            return RememberResult(updated, created=False, replaced=old.content)
        if await self._repo.count(user_id) >= self._max:
            raise LimitReachedError(f"user has {self._max} memories",
                                    public_message="You have too many saved memories. Delete "
                                                   "a few you no longer need first.")
        memory = Memory(id=uuid.uuid4(), user_id=user_id, kind=kind, content=text,
                        created_at=now, updated_at=now)
        await self._repo.add(memory, vector)
        logger.info("memory stored", extra={"kind": kind.value, "memory_id": str(memory.id)})
        return RememberResult(memory, created=True)

    async def _previous_version(self, user_id: str, kind: MemoryKind, vector: list[float],
                                replaces: uuid.UUID | None) -> Memory | None:
        """The memory a new one supersedes, if any (see :meth:`remember`).

        Raises:
            NotFoundError: ``replaces`` is not one of the user's memories.
        """
        if replaces is not None:
            memory = await self._repo.get(user_id, replaces)
            if memory is not None:
                return memory
            raise NotFoundError(f"memory {replaces} not found",
                                public_message="The memory to update doesn't exist.")
        if _is_zero(vector):
            return None
        similar = await self._repo.search(user_id, vector, limit=1, kind=kind)
        if similar and similar[0].similarity >= self._duplicate:
            return similar[0].memory
        return None

    async def recall(self, user_id: str, message: str, *, limit: int = 6) -> list[Memory]:
        """Memories to show the model for this message.

        Args:
            user_id: Owner.
            message: The user's new message.
            limit: Most relevant facts to add (goals and preferences come on top,
                up to ``MAX_ALWAYS_RECALLED``).

        Returns:
            Goals and preferences (newest first), then relevant facts (best first).

        Raises:
            StorageUnavailableError: The store failed.
        """
        standing: list[Memory] = []
        for kind in ALWAYS_RECALLED:
            standing.extend(await self._repo.all(user_id, kind))
        standing.sort(key=lambda m: m.updated_at, reverse=True)
        chosen = standing[:MAX_ALWAYS_RECALLED]
        if limit <= 0:
            return chosen
        [vector] = await self._embedder.embed([message])
        if _is_zero(vector):
            return chosen
        hits = await self._repo.search(user_id, vector, limit=limit, kind=MemoryKind.FACT)
        chosen.extend(hit.memory for hit in hits if hit.similarity >= self._relevance)
        return chosen

    async def all(self, user_id: str, kind: MemoryKind | None = None) -> list[Memory]:
        """Every memory of the user (newest first).

        Raises:
            StorageUnavailableError: The store failed.
        """
        return await self._repo.all(user_id, kind)

    async def forget(self, user_id: str, memory_id: uuid.UUID) -> None:
        """Delete a memory.

        Args:
            user_id: Owner.
            memory_id: The memory.

        Raises:
            NotFoundError: The user has no such memory.
            StorageUnavailableError: The store failed.
        """
        if not await self._repo.delete(user_id, memory_id):
            raise NotFoundError(f"memory {memory_id} not found",
                                public_message="That memory doesn't exist (or was already "
                                               "deleted).")
        logger.info("memory deleted", extra={"memory_id": str(memory_id)})


def _is_zero(vector: list[float]) -> bool:
    return not any(not math.isclose(v, 0.0) for v in vector)
