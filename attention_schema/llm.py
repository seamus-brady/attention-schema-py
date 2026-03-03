
from __future__ import annotations

import os
from typing import Protocol


class LLMClientProtocol(Protocol):
    def generate(self, system: str, messages: list[dict]) -> str: ...


class LLMClient:
    """Thin wrapper around the Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929") -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model

    def generate(self, system: str, messages: list[dict]) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return message.content[0].text


class MockLLMClient:
    """Deterministic stand-in for testing — echoes back the user message."""

    def __init__(self) -> None:
        self.last_system: str | None = None
        self.last_messages: list[dict] | None = None
        self.last_awareness_claims = None
        self.call_count: int = 0

    def generate(self, system: str, messages: list[dict]) -> str:
        self.last_system = system
        self.last_messages = messages
        self.call_count += 1
        # Extract the last user message for backward-compatible response format
        last_user = messages[-1]["content"] if messages else ""
        return f"[mock response to: {last_user}]"
