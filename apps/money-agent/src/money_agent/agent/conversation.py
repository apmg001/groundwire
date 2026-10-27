"""The conversation use case: the user sends a chat message.

    load history ─► recall memories ─► build prompt ─► agent loop ─► store both messages

* **History** is the last ``history_limit`` user/assistant messages of this
  conversation. Tool traffic is not stored or replayed: figures are always
  fetched fresh by tools, so old tool results cannot go stale in the prompt.
* **Memory** crosses conversations: goals and preferences always, plus facts
  relevant to the new message (see :class:`MemoryService`).
* **Atomic from the user's view**: both messages are stored only after the
  agent has answered. A failure stores nothing, so a retry starts clean.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from money_agent.agent.prompt import build_system_prompt
from money_agent.agent.runner import AgentRunner, ToolTrace, Verification
from money_agent.agent.tools import ToolContext
from money_agent.core.errors import InvalidInputError, NotFoundError
from money_agent.domain.models import ChatMessage, MessageRole
from money_agent.domain.ports import Clock, ConversationRepository, UserRepository
from money_agent.memory.service import MemoryService
from money_agent.services.finance import DataRange, FinanceService, NoDataError

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4_000


@dataclass(frozen=True, slots=True)
class ChatReply:
    """What the user gets back.

    Attributes:
        conversation_id: Conversation (new or continued).
        answer: The assistant's answer.
        verification: The answer check's outcome.
        tools: Tools the agent used.
        memories_used: Memories that were in the prompt.
    """

    conversation_id: uuid.UUID
    answer: str
    verification: Verification
    tools: tuple[ToolTrace, ...]
    memories_used: int


class ChatService:
    """Handles one chat message end to end."""

    def __init__(self, users: UserRepository, conversations: ConversationRepository,
                 memory: MemoryService, finance: FinanceService, runner: AgentRunner,
                 clock: Clock, *, history_limit: int = 20, recall_limit: int = 6) -> None:
        """Create the service.

        Args:
            users: User storage.
            conversations: Chat history storage.
            memory: Long-term memory.
            finance: Finance queries (for the data range in the prompt).
            runner: Agent loop.
            clock: Time source.
            history_limit: Earlier messages sent with each new one.
            recall_limit: Relevant facts added to the prompt.
        """
        self._users = users
        self._conversations = conversations
        self._memory = memory
        self._finance = finance
        self._runner = runner
        self._clock = clock
        self._history_limit = history_limit
        self._recall_limit = recall_limit

    async def send(self, user_id: str, text: str,
                   conversation_id: uuid.UUID | None = None) -> ChatReply:
        """Answer a message.

        Args:
            user_id: Sender.
            text: The message.
            conversation_id: Continue this conversation (new one when ``None``).

        Returns:
            The reply.

        Raises:
            InvalidInputError: Empty or too-long message.
            NotFoundError: ``conversation_id`` is not one of the user's.
            AssistantUnavailableError: The gateway failed.
            MessageRefusedError: The gateway refused the message.
            AgentError: The agent could not finish.
            StorageUnavailableError: The store failed.
        """
        message = text.strip()
        if not message or len(message) > MAX_MESSAGE_CHARS:
            raise InvalidInputError("bad message length", public_message=(
                f"Messages must be 1 to {MAX_MESSAGE_CHARS} characters."))
        await self._users.ensure(user_id)
        if conversation_id is None:
            conversation_id = await self._conversations.create(user_id)
            history: list[ChatMessage] = []
        elif not await self._conversations.exists(user_id, conversation_id):
            raise NotFoundError(f"conversation {conversation_id} not found",
                                public_message="That conversation doesn't exist.")
        else:
            history = await self._conversations.recent(user_id, conversation_id,
                                                       self._history_limit)

        memories = await self._memory.recall(user_id, message, limit=self._recall_limit)
        system = build_system_prompt(today=self._clock.today(), memories=memories,
                                     data=await self._data_range(user_id))
        messages = [{"role": "system", "content": system},
                    *({"role": m.role.value, "content": m.content} for m in history),
                    {"role": "user", "content": message}]
        received_at = self._clock.now()
        run = await self._runner.run(ToolContext(user_id=user_id), messages)

        meta = {"verification": run.verification.as_dict(),
                "tools": [t.name for t in run.tools]}
        await self._conversations.append(user_id, conversation_id, [
            ChatMessage(MessageRole.USER, message, received_at),
            ChatMessage(MessageRole.ASSISTANT, run.answer, self._clock.now(), meta)])
        return ChatReply(conversation_id=conversation_id, answer=run.answer,
                         verification=run.verification, tools=run.tools,
                         memories_used=len(memories))

    async def history(self, user_id: str, conversation_id: uuid.UUID,
                      limit: int = 50) -> list[ChatMessage]:
        """A conversation's recent messages, oldest first.

        Raises:
            NotFoundError: Not one of the user's conversations.
            StorageUnavailableError: The store failed.
        """
        if not await self._conversations.exists(user_id, conversation_id):
            raise NotFoundError(f"conversation {conversation_id} not found",
                                public_message="That conversation doesn't exist.")
        return await self._conversations.recent(user_id, conversation_id, limit)

    async def _data_range(self, user_id: str) -> DataRange | None:
        try:
            return await self._finance.data_range(user_id)
        except NoDataError:
            return None
