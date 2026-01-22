"""Session state models for Codex provider.

Follows the amplifier-claude branch pattern for disk-persisted sessions.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SessionMetadata(BaseModel):
    """Metadata about a session."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    codex_session_id: str | None = Field(default=None)
    name: str = Field(default="unnamed-session")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    turns: int = Field(default=0)
    total_tokens: int = Field(default=0)
    cache_read_tokens: int = Field(default=0)
    cache_creation_tokens: int = Field(default=0)
    cost_usd: float = Field(default=0.0)
    duration_seconds: float = Field(default=0.0)
    tags: list[str] = Field(default_factory=list)

    def update(self) -> None:
        """Update the timestamp."""
        self.updated_at = datetime.now()


class SessionState(BaseModel):
    """Complete session state."""

    metadata: SessionMetadata
    messages: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)

    def add_message(
        self, role: str, content: str, metadata: dict | None = None
    ) -> None:
        """Add a message to the session."""
        message: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            message["metadata"] = metadata
        self.messages.append(message)
        self.metadata.turns += 1
        self.metadata.update()

    def update_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read: int = 0,
        cache_creation: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
    ) -> None:
        """Update usage metrics for the session."""
        self.metadata.total_tokens += input_tokens + output_tokens
        self.metadata.cache_read_tokens += cache_read
        self.metadata.cache_creation_tokens += cache_creation
        self.metadata.cost_usd += cost_usd
        self.metadata.duration_seconds += duration_ms / 1000.0
        self.metadata.update()

    def set_codex_session_id(self, codex_session_id: str) -> None:
        """Set the Codex CLI session ID for caching."""
        self.metadata.codex_session_id = codex_session_id
        self.metadata.update()

    def get_cache_efficiency(self) -> float:
        """Calculate cache efficiency ratio."""
        if self.metadata.total_tokens == 0:
            return 0.0
        return self.metadata.cache_read_tokens / self.metadata.total_tokens
