"""Command line entry point.

    python -m agent_eval validate [tasks_dir]
    python -m agent_eval run [tasks_dir] [--llm fake|openai] [--task ID ...] [--out FILE]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from agent_eval.llm import FakeLLM, LLMError, OpenAIClient
from agent_eval.runner import LLMAgent, run
from agent_eval.tasks import load_tasks
from agent_eval.validate import validate


def cmd_validate(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    by_category = Counter(t.category.value for t in tasks)
    print(f"{len(tasks)} tasks: " + ", ".join(f"{n} {c}" for c, n in sorted(by_category.items())))

    failures = validate(args.tasks)
    for task_id, problems in failures.items():
        for p in problems:
            print(f"FAIL {task_id}: {p}")
    print("all tasks valid" if not failures else f"{len(failures)} task(s) invalid")
    return 1 if failures else 0


def cmd_run(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    if args.task:
        unknown = set(args.task) - {t.id for t in tasks}
        if unknown:
            print(f"unknown task ids: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        tasks = [t for t in tasks if t.id in args.task]
    try:
        llm = FakeLLM.from_tasks(tasks) if args.llm == "fake" else OpenAIClient.from_env()
    except LLMError as e:
        print(e, file=sys.stderr)
        return 2
    out = args.out or Path("runs") / f"{llm.name.replace('/', '_')}.jsonl"

    traces = run(LLMAgent(llm), tasks, out=out)
    statuses = Counter(t.status for t in traces)
    print(f"{len(traces)} tasks: " + ", ".join(f"{n} {s}" for s, n in sorted(statuses.items())))
    print(f"steps: {sum(len(t.steps) for t in traces)}")
    note = " (estimated, fake LLM)" if args.llm == "fake" else ""
    print(f"tokens in/out: {sum(t.input_tokens for t in traces)}/{sum(t.output_tokens for t in traces)}{note}")
    print(f"wall time: {sum(t.wall_time_s for t in traces):.2f}s")
    print(f"trace: {out}")
    return 1 if statuses["llm_error"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_eval")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="check that every task is well-formed and answerable")
    v.add_argument("tasks", nargs="?", type=Path, default=Path("tasks"))
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="run an agent on the tasks and save a JSONL trace")
    r.add_argument("tasks", nargs="?", type=Path, default=Path("tasks"))
    r.add_argument("--llm", choices=["fake", "openai"], default="fake", help="fake needs no API key")
    r.add_argument("--task", action="append", metavar="ID", help="only run this task (repeatable)")
    r.add_argument("--out", type=Path, help="trace file (default: runs/<model>.jsonl)")
    r.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
