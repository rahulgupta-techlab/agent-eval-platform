"""Checks that the benchmark itself is sound before any agent is scored against it.

For every task, the expected tool calls are replayed against a fresh environment.
They must run without errors, and their outputs must contain the expected answer.
That rules out gold answers that no agent could ground in the tools.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent_eval.tasks import AnswerType, Task, load_tasks
from agent_eval.tools import Environment, ToolError

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def check_task(task: Task) -> list[str]:
    """Return a list of problems with one task (empty if it is sound)."""
    env = Environment()
    problems: list[str] = []
    outputs: list[str] = []
    last_search: list[str] | None = None

    for call in task.expected_calls:
        tool = env.tools.get(call.tool)
        if tool is None:
            problems.append(f"unknown tool {call.tool!r}")
            continue
        params = tool.parameters
        missing = set(params.get("required", [])) - call.args.keys()
        extra = call.args.keys() - params["properties"].keys()
        if missing or extra:
            problems.append(f"{call.tool}: missing args {sorted(missing)}, unexpected args {sorted(extra)}")
            continue
        try:
            result = env.call(call.tool, call.args)
        except ToolError as e:
            problems.append(f"{call.tool} failed: {e}")
            continue
        if call.tool == "search_docs":
            last_search = [hit["doc_id"] for hit in result]
        if call.tool == "read_doc" and last_search is not None and call.args["doc_id"] not in last_search:
            problems.append(f"search_docs did not surface {call.args['doc_id']!r} (got {last_search})")
        outputs.append(result if isinstance(result, str) else json.dumps(result))

    evidence = "\n".join(outputs).lower()
    answer = task.expected_answer
    if answer.type is AnswerType.CONTAINS:
        for options in answer.alternatives():
            if not any(o.lower() in evidence for o in options):
                problems.append(f"answer {options} not found in tool outputs")
    elif answer.type is AnswerType.NUMERIC:
        numbers = [float(n) for n in NUMBER.findall(evidence)]
        if not any(abs(n - answer.value) <= answer.tolerance for n in numbers):
            problems.append(f"answer {answer.value} not found in tool outputs")
    return problems


def validate(path: Path) -> dict[str, list[str]]:
    """Validate every task under `path`. Maps task id to its problems (only failing tasks)."""
    report = {t.id: check_task(t) for t in load_tasks(path)}
    return {task_id: p for task_id, p in report.items() if p}
