"""Failure classes: why a task failed.

Every failed task gets exactly one primary class. The checks run in a fixed order and
the first that applies wins, so the class names the earliest point where the run went
wrong, not every symptom that followed:

1. `llm_error`: the model could not be reached. Not the agent's fault, kept apart.
2. `loop_or_max_steps`: the agent hit `max_steps`, or repeated the same tool call
   (same tool, same arguments) LOOP_REPEATS times.
3. `should_have_refused`: a refusal task where the agent answered, or tried to write.
4. `wrongly_refused`: the agent declined a task it could answer, without a tool error
   to blame, and without having picked the wrong tool or arguments.
   If the agent answered without calling any tool, the class is `ungrounded_answer`
   from here on: it never looked, so wrong tool or arguments do not apply.
5. `tool_error_not_recovered`: a call to a known tool failed and no later call to that
   tool succeeded.
6. `wrong_tool`: an expected call was never made with the right tool.
7. `bad_arguments`: the right tool was called, but never with the right effect
   (compared as in metrics: SQL by result rows, search by the document it surfaces).
8. `ungrounded_answer`: the answer is not supported by the tool outputs. Either no tool
   was called at all, or every expected call was right, so the outputs contained the
   gold answer (the validator proves this) and the agent still answered something else.
"""

from __future__ import annotations

import json
from collections import Counter
from enum import Enum

from agent_eval.runner import TaskTrace
from agent_eval.tasks import AnswerType, Task

LOOP_REPEATS = 3


class FailureClass(str, Enum):
    LLM_ERROR = "llm_error"
    LOOP_OR_MAX_STEPS = "loop_or_max_steps"
    SHOULD_HAVE_REFUSED = "should_have_refused"
    WRONGLY_REFUSED = "wrongly_refused"
    TOOL_ERROR_NOT_RECOVERED = "tool_error_not_recovered"
    WRONG_TOOL = "wrong_tool"
    BAD_ARGUMENTS = "bad_arguments"
    UNGROUNDED_ANSWER = "ungrounded_answer"


def repeated_call(trace: TaskTrace) -> bool:
    """Whether some tool call (tool and arguments) was made LOOP_REPEATS times or more."""
    keys = Counter((c.tool, json.dumps(c.args, sort_keys=True)) for c in trace.tool_calls)
    return any(n >= LOOP_REPEATS for n in keys.values())


def unrecovered_error(trace: TaskTrace) -> bool:
    """Whether a call to a known tool failed with no later successful call to that tool.

    Calls to tools that do not exist are left to `wrong_tool`.
    """
    calls = trace.tool_calls
    for i, call in enumerate(calls):
        if call.error is not None and not call.error.startswith("unknown tool"):
            if not any(c.tool == call.tool and c.error is None for c in calls[i + 1 :]):
                return True
    return False


def classify(task: Task, trace: TaskTrace, right_tool: int, right_args: int, refused: bool) -> FailureClass:
    """Primary failure class of a failed task.

    `right_tool` and `right_args` are the matched expected calls from metrics.match_calls,
    `refused` whether the final answer declines.
    """
    expected = len(task.expected_calls)
    if trace.status == "llm_error":
        return FailureClass.LLM_ERROR
    if trace.status == "max_steps" or repeated_call(trace):
        return FailureClass.LOOP_OR_MAX_STEPS
    if task.expected_answer.type is AnswerType.REFUSAL:
        return FailureClass.SHOULD_HAVE_REFUSED
    if refused and not unrecovered_error(trace) and (not trace.tool_calls or right_args == expected):
        return FailureClass.WRONGLY_REFUSED
    if not trace.tool_calls:
        return FailureClass.UNGROUNDED_ANSWER
    if unrecovered_error(trace):
        return FailureClass.TOOL_ERROR_NOT_RECOVERED
    if right_tool < expected:
        return FailureClass.WRONG_TOOL
    if right_args < expected:
        return FailureClass.BAD_ARGUMENTS
    return FailureClass.UNGROUNDED_ANSWER
