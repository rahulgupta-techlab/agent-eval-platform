"""Runs an agent on benchmark tasks and records what happened at every step.

Each task gets a fresh Environment. One step is one model call plus the tool calls it
asked for. The run stops when the model answers without calling a tool, or after the
task's `max_steps`. Traces are written as JSONL, one task per line.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, Field

from agent_eval.llm import LLM, LLMError, LLMResponse, ToolCall
from agent_eval.tasks import Task
from agent_eval.tools import Environment, ToolError

SYSTEM_PROMPT = (
    "You are a support assistant for an outdoor-gear shop. Use the tools to look up facts "
    "in the shop database and policy documents; do not guess. The data is read-only: decline "
    "requests to change it. If the data does not contain the answer, say so. When you are "
    "done, reply with a short final answer and no tool calls."
)


class Agent(Protocol):
    """Something that can work on a task: builds the opening messages, then picks each next step."""

    name: str

    def start(self, task: Task) -> list[dict[str, Any]]: ...

    def act(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class LLMAgent:
    """Plain tool-calling loop: a system prompt and an LLM."""

    def __init__(self, llm: LLM, system_prompt: str = SYSTEM_PROMPT, name: str | None = None) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.name = name or llm.name

    def start(self, task: Task) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": task.prompt}]

    def act(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        return self.llm.complete(messages, tools)


class ToolCallRecord(BaseModel):
    id: str
    tool: str
    args: dict[str, Any] | None  # None if the arguments were not a JSON object
    result: Any = None
    error: str | None = None


class StepTrace(BaseModel):
    step: int
    messages: list[dict[str, Any]]  # messages added in this step; the first step includes the prompt
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0


class TaskTrace(BaseModel):
    task_id: str
    agent: str
    status: Literal["answered", "max_steps", "llm_error"]
    answer: str | None = None
    error: str | None = None
    steps: list[StepTrace] = Field(default_factory=list)

    @property
    def tool_calls(self) -> list[ToolCallRecord]:
        return [c for s in self.steps for c in s.tool_calls]

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def wall_time_s(self) -> float:
        return sum(s.wall_time_s for s in self.steps)


def execute(env: Environment, call: ToolCall) -> ToolCallRecord:
    """Run one tool call. Errors are recorded, not raised, so the agent can recover."""
    try:
        args = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as e:
        return ToolCallRecord(id=call.id, tool=call.name, args=None, error=f"arguments are not valid JSON: {e}")
    if not isinstance(args, dict):
        return ToolCallRecord(id=call.id, tool=call.name, args=None, error="arguments must be a JSON object")
    try:
        return ToolCallRecord(id=call.id, tool=call.name, args=args, result=env.call(call.name, args))
    except ToolError as e:
        return ToolCallRecord(id=call.id, tool=call.name, args=args, error=str(e))


def tool_message(record: ToolCallRecord) -> dict[str, Any]:
    if record.error is not None:
        content = f"Error: {record.error}"
    elif isinstance(record.result, str):
        content = record.result
    else:
        content = json.dumps(record.result, ensure_ascii=False)
    return {"role": "tool", "tool_call_id": record.id, "content": content}


def run_task(agent: Agent, task: Task, env_factory: Callable[[], Environment] = Environment) -> TaskTrace:
    env = env_factory()
    tools = [t.schema() for t in env.tools.values()]
    messages = agent.start(task)
    pending = list(messages)  # messages not yet attached to a step
    trace = TaskTrace(task_id=task.id, agent=agent.name, status="max_steps")

    for step in range(task.max_steps):
        start = time.perf_counter()
        try:
            response = agent.act(messages, tools)
        except LLMError as e:
            trace.status, trace.error = "llm_error", str(e)
            break
        new = [response.message()]
        records = [execute(env, call) for call in response.tool_calls]
        new += [tool_message(r) for r in records]
        messages += new
        trace.steps.append(
            StepTrace(
                step=step,
                messages=pending + new,
                tool_calls=records,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                wall_time_s=round(time.perf_counter() - start, 6),
            )
        )
        pending = []
        if not response.tool_calls:
            trace.status, trace.answer = "answered", response.content or ""
            break
    return trace


def run(
    agent: Agent,
    tasks: list[Task],
    out: Path | None = None,
    env_factory: Callable[[], Environment] = Environment,
) -> list[TaskTrace]:
    """Run `agent` on every task, each in a fresh environment. Writes JSONL to `out` as it goes."""
    traces = []
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
    for task in tasks:
        trace = run_task(agent, task, env_factory)
        traces.append(trace)
        if out is not None:
            with out.open("a", encoding="utf-8") as f:
                f.write(trace.model_dump_json() + "\n")
    return traces


def load_traces(path: Path) -> list[TaskTrace]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [TaskTrace.model_validate_json(line) for line in lines if line.strip()]
