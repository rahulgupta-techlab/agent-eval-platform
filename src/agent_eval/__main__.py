"""Command line entry point.

    python -m agent_eval validate [tasks_dir]
    python -m agent_eval run [tasks_dir] [--llm fake|openai] [--task ID ...] [--out FILE]
    python -m agent_eval score TRACE [--tasks DIR] [--prices FILE] [--json FILE]
    python -m agent_eval compare [CONFIG] [--tasks DIR] [--run NAME ...] [--out-dir DIR]
                                 [--baseline FILE] [--threshold X]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from agent_eval.compare import (
    check_regression,
    compare,
    load_comparison,
    load_config,
    markdown_table,
    write_results,
)
from agent_eval.llm import FakeLLM, LLMError, OpenAIClient
from agent_eval.metrics import PRICES_FILE, load_prices, score_run, summary
from agent_eval.runner import LLMAgent, load_traces, run
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
    print(summary(score_run(tasks, traces, load_prices())))
    if args.llm == "fake":
        print("note: fake LLM, so token counts are estimates and the results say nothing about a model")
    print(f"trace: {out}")
    return 1 if statuses["llm_error"] else 0


def cmd_score(args: argparse.Namespace) -> int:
    try:
        metrics = score_run(load_tasks(args.tasks), load_traces(args.trace), load_prices(args.prices))
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    print(summary(metrics))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(metrics.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"metrics: {args.json}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        baseline = load_comparison(args.baseline) if args.baseline else None
    except (OSError, ValueError, ValidationError) as e:
        print(e, file=sys.stderr)
        return 2
    if args.run:
        unknown = set(args.run) - {r.name for r in config.runs}
        if unknown:
            print(f"unknown runs: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        config.runs = [r for r in config.runs if r.name in args.run]
    try:
        comparison = compare(config, load_tasks(args.tasks), load_prices(args.prices), out_dir=args.out_dir)
    except LLMError as e:
        print(e, file=sys.stderr)
        return 2
    print(markdown_table(comparison))
    if any(r.llm == "fake" for r in comparison.runs):
        print("note: fake runs are scripted, so their results say nothing about a model")
    md, js = write_results(comparison, args.out_dir)
    print(f"results: {md}, {js}")

    status = 1 if any(r.metrics.failures.get("llm_error") for r in comparison.runs) else 0
    if baseline is not None:
        checks = check_regression(comparison, baseline, args.threshold)
        for c in checks:
            print(c.line())
        if not checks:
            print("regression check: no run in common with the baseline", file=sys.stderr)
            return 1
        if any(c.regressed for c in checks):
            print(f"regression: success dropped by more than {args.threshold:.1%}")
            return 1
    return status


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

    s = sub.add_parser("score", help="score a saved trace: success, tool-call accuracy, latency, cost")
    s.add_argument("trace", type=Path)
    s.add_argument("--tasks", type=Path, default=Path("tasks"))
    s.add_argument("--prices", type=Path, default=PRICES_FILE, help="price table (USD per 1M tokens)")
    s.add_argument("--json", type=Path, help="also write per-task and run metrics as JSON")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare", help="run several model x prompt configs on the same tasks and compare")
    c.add_argument("config", nargs="?", type=Path, default=Path("config") / "compare.yaml")
    c.add_argument("--tasks", type=Path, default=Path("tasks"))
    c.add_argument("--run", action="append", metavar="NAME", help="only this run from the config (repeatable)")
    c.add_argument("--out-dir", type=Path, default=Path("runs") / "compare", help="traces and results")
    c.add_argument("--prices", type=Path, default=PRICES_FILE, help="price table (USD per 1M tokens)")
    c.add_argument("--baseline", type=Path, help="saved results.json to check for regressions against")
    c.add_argument("--threshold", type=float, default=0.05, help="allowed drop in success rate (default 0.05)")
    c.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
