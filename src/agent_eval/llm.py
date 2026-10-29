"""LLM backends an agent can run on.

`FakeLLM` is the default. It replays a fixed script per task, so tests and CI need no
API key and produce the same trace on every run. `OpenAIClient` talks to any
OpenAI-compatible /chat/completions endpoint (OpenAI, vLLM, Ollama, ...).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_eval.tasks import AnswerType, ExpectedAnswer, Task


class LLMError(Exception):
    """The model could not be reached or returned something unusable."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, exactly as the model produced it


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def message(self) -> dict[str, Any]:
        """The response as an OpenAI-style assistant message."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return msg


class LLM(Protocol):
    name: str

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


@dataclass
class Turn:
    """One scripted model reply: tool calls, or a final answer if there are none."""

    content: str | None = None
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def estimate_tokens(obj: Any) -> int:
    """Rough token count (4 characters per token), used only by FakeLLM."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return max(1, len(text) // 4)


class FakeLLM:
    """Deterministic model that replays a script, keyed by the task prompt.

    The n-th call in a conversation returns the n-th turn of the script for that
    conversation's user prompt. Past the end of the script, or for an unknown prompt,
    it gives up with a final answer. Token counts are estimates, not real usage.
    """

    def __init__(self, scripts: dict[str, list[Turn]], name: str = "fake") -> None:
        self.scripts = scripts
        self.name = name

    @classmethod
    def from_tasks(cls, tasks: list[Task], name: str = "fake", give_up_every: int = 0) -> FakeLLM:
        """A perfect agent: makes each task's expected calls, then states the gold answer.

        With `give_up_every=n`, every n-th task (1-based, in the order given) is scripted to
        answer at once without calling a tool, which gives a deterministically worse agent.
        """
        scripts = {
            t.prompt: [Turn(content="I could not find the answer.")]
            if give_up_every and i % give_up_every == 0
            else [Turn(calls=[(c.tool, c.args)]) for c in t.expected_calls]
            + [Turn(content=gold_answer(t.expected_answer))]
            for i, t in enumerate(tasks, start=1)
        }
        return cls(scripts, name=name)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
        turn_no = sum(1 for m in messages if m["role"] == "assistant")
        script = self.scripts.get(prompt, [])
        turn = script[turn_no] if turn_no < len(script) else Turn(content="I could not find the answer.")
        calls = [
            ToolCall(id=f"call_{turn_no}_{i}", name=tool, arguments=json.dumps(args))
            for i, (tool, args) in enumerate(turn.calls)
        ]
        response = LLMResponse(content=turn.content, tool_calls=calls)
        response.input_tokens = estimate_tokens(messages) + estimate_tokens(tools)
        response.output_tokens = estimate_tokens(response.message())
        return response


def gold_answer(answer: ExpectedAnswer) -> str:
    """A final answer that satisfies `answer`, used by the scripted perfect agent."""
    if answer.type is AnswerType.NUMERIC:
        return f"The answer is {answer.value:g}."
    if answer.type is AnswerType.REFUSAL:
        return "I can't do that: the shop data is read-only and does not contain this information."
    return "The answer: " + "; ".join(options[0] for options in answer.alternatives()) + "."


class OpenAIClient:
    """Client for an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    @classmethod
    def from_env(cls, model: str | None = None) -> OpenAIClient:
        """Configure from AGENT_EVAL_MODEL, AGENT_EVAL_BASE_URL and AGENT_EVAL_API_KEY.

        `model`, if given, takes the place of AGENT_EVAL_MODEL.
        """
        model = model or os.environ.get("AGENT_EVAL_MODEL")
        if not model:
            raise LLMError("set AGENT_EVAL_MODEL (and AGENT_EVAL_BASE_URL / AGENT_EVAL_API_KEY as needed)")
        return cls(
            model=model,
            base_url=os.environ.get("AGENT_EVAL_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("AGENT_EVAL_API_KEY", ""),
        )

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        body = {"model": self.model, "messages": messages, "tools": tools, "temperature": self.temperature}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code}: {e.read()[:500].decode(errors='replace')}") from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise LLMError(f"request failed: {e}") from e
        try:
            msg = data["choices"][0]["message"]
            calls = [
                ToolCall(id=c["id"], name=c["function"]["name"], arguments=c["function"].get("arguments") or "{}")
                for c in msg.get("tool_calls") or []
            ]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response: {json.dumps(data)[:500]}") from e
        usage = data.get("usage") or {}
        return LLMResponse(
            content=msg.get("content"),
            tool_calls=calls,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
