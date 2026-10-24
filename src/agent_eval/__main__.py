"""Command line entry point: `python -m agent_eval validate [tasks_dir]`."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from agent_eval.tasks import load_tasks
from agent_eval.validate import validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_eval")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="check that every task is well-formed and answerable")
    v.add_argument("tasks", nargs="?", type=Path, default=Path("tasks"))
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    by_category = Counter(t.category.value for t in tasks)
    print(f"{len(tasks)} tasks: " + ", ".join(f"{n} {c}" for c, n in sorted(by_category.items())))

    failures = validate(args.tasks)
    for task_id, problems in failures.items():
        for p in problems:
            print(f"FAIL {task_id}: {p}")
    print("all tasks valid" if not failures else f"{len(failures)} task(s) invalid")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
