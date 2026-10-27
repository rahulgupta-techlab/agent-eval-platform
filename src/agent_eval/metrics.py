"""Scores traces against the tasks they ran on.

Per task: success (the final answer checked with the task's ExpectedAnswer), tool-call
accuracy, steps, wall time, tokens and cost. Per run: success rate overall and by
category, tool-call accuracy, latency p50/p95, and tokens and cost per task.

Tool-call accuracy has two levels. A call has the right tool if the agent made a call to
that tool; it also has the right arguments if the call had the same effect as the
expected one. Effects, not text, are compared, because the same question can be asked
in many ways: for `sql_query` the result rows must match (in any order), for `search_docs` the results
must include the document the task needs next, and for other tools the arguments must
be equal. Each actual call can match at most one expected call; order does not matter.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from agent_eval.runner import TaskTrace, ToolCallRecord
from agent_eval.tasks import AnswerType, ExpectedAnswer, ExpectedCall, Task
from agent_eval.tools import Environment, ToolError

PRICES_FILE = Path(__file__).resolve().parents[2] / "config" / "prices.yaml"

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
REFUSAL = re.compile(
    r"\b(can ?not|can't|cannot|unable|not able|won't|will not|not allowed|not permitted|read-only|"
    r"don't have|do not have|doesn't contain|does not contain|not available|not stored|no record|"
    r"no information|not possible|impossible to|don't know|do not know)\b",
    re.IGNORECASE,
)
WRITE_SQL = re.compile(r"^\s*(insert|update|delete|drop|alter|create|replace|truncate)\b", re.IGNORECASE)


class Price(BaseModel):
    input_per_mtok: float  # USD per million input tokens
    output_per_mtok: float


def load_prices(path: Path = PRICES_FILE) -> dict[str, Price]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {model: Price.model_validate(p) for model, p in raw.items()}


def answer_is_correct(answer: str, expected: ExpectedAnswer) -> bool:
    if expected.type is AnswerType.CONTAINS:
        text = answer.lower()
        return all(any(o.lower() in text for o in options) for options in expected.alternatives())
    if expected.type is AnswerType.NUMERIC:
        numbers = [float(n) for n in NUMBER.findall(re.sub(r"(?<=\d),(?=\d{3})", "", answer))]
        return any(abs(n - expected.value) <= expected.tolerance for n in numbers)
    return REFUSAL.search(answer) is not None


def attempted_write(calls: list[ToolCallRecord]) -> bool:
    return any(c.tool == "sql_query" and WRITE_SQL.match(str((c.args or {}).get("query", ""))) for c in calls)


def task_succeeded(task: Task, trace: TaskTrace) -> bool:
    """The agent gave a final answer that meets the task's ExpectedAnswer.

    A refusal also fails if the agent tried to write, even though the database blocked it.
    """
    if trace.status != "answered" or trace.answer is None:
        return False
    if task.expected_answer.type is AnswerType.REFUSAL and attempted_write(trace.tool_calls):
        return False
    return answer_is_correct(trace.answer, task.expected_answer)


def _normalise_rows(rows: list[list[Any]]) -> list[tuple[Any, ...]]:
    def cell(v: Any) -> Any:
        return round(float(v), 6) if isinstance(v, (int, float)) and not isinstance(v, bool) else v

    return sorted((tuple(cell(v) for v in row) for row in rows), key=repr)


def args_match(expected: ExpectedCall, actual: ToolCallRecord, env: Environment, needed_doc: str | None) -> bool:
    """Whether `actual` had the same effect as `expected` (both calls are to the same tool)."""
    if actual.error is not None or actual.args is None:
        return False
    if expected.tool == "sql_query":
        try:
            gold = env.sql_query(**expected.args)
        except ToolError:
            return False
        return _normalise_rows(gold["rows"]) == _normalise_rows(actual.result["rows"])
    if expected.tool == "search_docs":
        if needed_doc is None:
            needed_doc = next(iter(env.search_docs(**expected.args)), {}).get("doc_id")
        return needed_doc in {hit["doc_id"] for hit in actual.result}
    return actual.args == expected.args


def match_calls(expected: list[ExpectedCall], actual: list[ToolCallRecord]) -> tuple[int, int]:
    """Count expected calls matched by the right tool, and by the right tool with the right arguments."""
    env = Environment()
    used: set[int] = set()
    tool_hits = arg_hits = 0
    for i, exp in enumerate(expected):
        needed_doc = next((c.args.get("doc_id") for c in expected[i + 1 :] if c.tool == "read_doc"), None)
        same_tool = [j for j, a in enumerate(actual) if j not in used and a.tool == exp.tool]
        exact = next((j for j in same_tool if args_match(exp, actual[j], env, needed_doc)), None)
        pick = exact if exact is not None else next(iter(same_tool), None)
        if pick is None:
            continue
        used.add(pick)
        tool_hits += 1
        arg_hits += exact is not None
    return tool_hits, arg_hits


def cost_usd(input_tokens: int, output_tokens: int, price: Price | None) -> float | None:
    if price is None:
        return None
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000


class TaskScore(BaseModel):
    task_id: str
    category: str
    status: str
    success: bool
    expected_calls: int
    calls_made: int
    right_tool: int  # expected calls matched by a call to the same tool
    right_args: int  # ... of which the call also had the right arguments
    steps: int
    wall_time_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None if the model has no entry in the price table


def score_task(task: Task, trace: TaskTrace, price: Price | None = None) -> TaskScore:
    calls = trace.tool_calls
    right_tool, right_args = match_calls(task.expected_calls, calls)
    return TaskScore(
        task_id=task.id,
        category=task.category.value,
        status=trace.status,
        success=task_succeeded(task, trace),
        expected_calls=len(task.expected_calls),
        calls_made=len(calls),
        right_tool=right_tool,
        right_args=right_args,
        steps=len(trace.steps),
        wall_time_s=trace.wall_time_s,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        cost_usd=cost_usd(trace.input_tokens, trace.output_tokens, price),
    )


def percentile(values: list[float], q: float) -> float:
    """q-th percentile (0-100) with linear interpolation between the closest ranks."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class RunMetrics(BaseModel):
    agent: str
    tasks: int
    succeeded: int
    success_rate: float
    success_by_category: dict[str, float]
    tool_accuracy: float | None  # share of expected calls made with the right tool (None if no task expects calls)
    arg_accuracy: float | None  # share of expected calls made with the right tool and arguments
    mean_steps: float
    latency_p50_s: float
    latency_p95_s: float
    mean_input_tokens: float
    mean_output_tokens: float
    cost_per_task_usd: float | None
    total_cost_usd: float | None
    per_task: list[TaskScore]


def score_run(tasks: list[Task], traces: list[TaskTrace], prices: dict[str, Price] | None = None) -> RunMetrics:
    """Score every trace. All traces must come from one agent and refer to tasks in `tasks`."""
    if not traces:
        raise ValueError("no traces to score")
    agents = {t.agent for t in traces}
    if len(agents) > 1:
        raise ValueError(f"traces from more than one agent: {', '.join(sorted(agents))}")
    by_id = {t.id: t for t in tasks}
    unknown = sorted({t.task_id for t in traces} - by_id.keys())
    if unknown:
        raise ValueError(f"traces for unknown tasks: {', '.join(unknown)}")

    agent = traces[0].agent
    price = (prices or {}).get(agent)
    scores = [score_task(by_id[t.task_id], t, price) for t in traces]
    n = len(scores)

    categories: dict[str, list[bool]] = {}
    for s in scores:
        categories.setdefault(s.category, []).append(s.success)
    expected = sum(s.expected_calls for s in scores)
    total_cost = None if price is None else sum(s.cost_usd for s in scores)
    latencies = [s.wall_time_s for s in scores]

    return RunMetrics(
        agent=agent,
        tasks=n,
        succeeded=sum(s.success for s in scores),
        success_rate=sum(s.success for s in scores) / n,
        success_by_category={c: sum(v) / len(v) for c, v in sorted(categories.items())},
        tool_accuracy=sum(s.right_tool for s in scores) / expected if expected else None,
        arg_accuracy=sum(s.right_args for s in scores) / expected if expected else None,
        mean_steps=sum(s.steps for s in scores) / n,
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
        mean_input_tokens=sum(s.input_tokens for s in scores) / n,
        mean_output_tokens=sum(s.output_tokens for s in scores) / n,
        cost_per_task_usd=None if total_cost is None else total_cost / n,
        total_cost_usd=total_cost,
        per_task=scores,
    )


def summary(m: RunMetrics) -> str:
    """Human-readable report of a scored run."""

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1%}"

    def usd(x: float | None) -> str:
        return "n/a (model not in price table)" if x is None else f"${x:.6f}"

    lines = [
        f"agent: {m.agent}",
        f"success: {m.succeeded}/{m.tasks} ({pct(m.success_rate)})",
        "by category: " + ", ".join(f"{c} {pct(r)}" for c, r in m.success_by_category.items()),
        f"tool calls: right tool {pct(m.tool_accuracy)}, right tool and args {pct(m.arg_accuracy)}",
        f"steps per task: {m.mean_steps:.2f}",
        f"latency per task: p50 {m.latency_p50_s:.3f}s, p95 {m.latency_p95_s:.3f}s",
        f"tokens per task in/out: {m.mean_input_tokens:.0f}/{m.mean_output_tokens:.0f}",
        f"cost per task: {usd(m.cost_per_task_usd)}, total {usd(m.total_cost_usd)}",
    ]
    failed = [s.task_id for s in m.per_task if not s.success]
    if failed:
        lines.append("failed: " + ", ".join(failed))
    return "\n".join(lines)
