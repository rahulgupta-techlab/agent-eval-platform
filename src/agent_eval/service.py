"""HTTP service for starting runs and reading their results.

    POST /runs              start a run (one model x prompt config) on all or some tasks
    GET  /runs              every run so far, newest first
    GET  /runs/{id}         status, and the metrics once the run is done
    GET  /runs/{id}/report  HTML report of a finished run

Runs execute in the background, one after another, and are kept in memory; each trace is
also saved as <runs_dir>/<id>.jsonl. Start it with `python -m agent_eval serve`.
"""

from __future__ import annotations

import html
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agent_eval.compare import RunConfig, build_agent
from agent_eval.metrics import PRICES_FILE, RunMetrics, load_prices, score_run
from agent_eval.runner import SYSTEM_PROMPT, TaskTrace, run
from agent_eval.tasks import Task, load_tasks


class RunRequest(BaseModel):
    name: str = Field("run", pattern=r"^[A-Za-z0-9._-]+$")
    llm: Literal["fake", "openai"] = "fake"
    model: str | None = None  # required for openai
    system_prompt: str | None = None  # default: the runner's SYSTEM_PROMPT
    give_up_every: int = Field(0, ge=0)  # fake only: every n-th task answers without tools
    task_ids: list[str] | None = None  # default: every task


class RunStatus(BaseModel):
    id: str
    name: str
    llm: str
    model: str
    status: Literal["queued", "running", "done", "failed"]
    created_at: str
    tasks: int
    error: str | None = None
    metrics: RunMetrics | None = None


class _Run:
    def __init__(self, status: RunStatus, config: RunConfig, prompts: dict[str, str], tasks: list[Task]) -> None:
        self.status = status
        self.config = config
        self.prompts = prompts
        self.tasks = tasks
        self.traces: list[TaskTrace] = []


def create_app(
    tasks_dir: Path = Path("tasks"), runs_dir: Path = Path("runs") / "service", prices_file: Path = PRICES_FILE
) -> FastAPI:
    tasks = load_tasks(tasks_dir)
    prices = load_prices(prices_file)
    runs: dict[str, _Run] = {}
    lock = threading.Lock()  # one run at a time: runs share nothing, but this keeps latencies comparable

    app = FastAPI(title="Agent Eval Platform")

    def execute(r: _Run) -> None:
        with lock:
            r.status.status = "running"
            try:
                agent = build_agent(r.config, r.prompts, r.tasks)
                r.traces = run(agent, r.tasks, out=runs_dir / f"{r.status.id}.jsonl")
                price = prices.get(r.config.model_name)
                r.status.metrics = score_run(r.tasks, r.traces, {r.config.name: price} if price else None)
                r.status.status = "done"
            except Exception as e:  # reported through GET /runs/{id}, not raised into the server
                r.status.status, r.status.error = "failed", f"{type(e).__name__}: {e}"

    def get(run_id: str) -> _Run:
        if run_id not in runs:
            raise HTTPException(404, f"no run {run_id}")
        return runs[run_id]

    @app.post("/runs", status_code=202, response_model=RunStatus)
    def start_run(req: RunRequest, background: BackgroundTasks) -> RunStatus:
        try:
            config = RunConfig(
                name=req.name,
                llm=req.llm,
                model=req.model,
                prompt="default" if req.system_prompt is None else "custom",
                give_up_every=req.give_up_every,
            )
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        selected = tasks
        if req.task_ids is not None:
            unknown = sorted(set(req.task_ids) - {t.id for t in tasks})
            if unknown or not req.task_ids:
                raise HTTPException(422, f"unknown task ids: {', '.join(unknown)}" if unknown else "no task ids")
            selected = [t for t in tasks if t.id in req.task_ids]
        prompts = {"default": SYSTEM_PROMPT}
        if req.system_prompt is not None:
            prompts["custom"] = req.system_prompt

        status = RunStatus(
            id=uuid.uuid4().hex[:12],
            name=config.name,
            llm=config.llm,
            model=config.model_name,
            status="queued",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tasks=len(selected),
        )
        r = _Run(status, config, prompts, selected)
        runs[status.id] = r
        background.add_task(execute, r)
        return status

    @app.get("/runs", response_model=list[RunStatus])
    def list_runs() -> list[RunStatus]:
        """Without metrics, which can be large; GET /runs/{id} has them."""
        return [r.status.model_copy(update={"metrics": None}) for r in reversed(runs.values())]

    @app.get("/runs/{run_id}", response_model=RunStatus)
    def run_status(run_id: str) -> RunStatus:
        return get(run_id).status

    @app.get("/runs/{run_id}/report", response_class=HTMLResponse)
    def run_report(run_id: str) -> str:
        r = get(run_id)
        if r.status.status != "done":
            raise HTTPException(409, f"run {run_id} is {r.status.status}, the report needs a finished run")
        return report_html(r.status, r.tasks, r.traces)

    return app


def report_html(status: RunStatus, tasks: list[Task], traces: list[TaskTrace]) -> str:
    """A self-contained HTML page: run metrics, then every task with its tool calls."""
    m = status.metrics
    assert m is not None
    e = html.escape

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1%}"

    def usd(x: float | None) -> str:
        return "n/a" if x is None else f"${x:.6f}"

    overview = [
        ("Success", f"{m.succeeded}/{m.tasks} ({pct(m.success_rate)})"),
        ("Right tool", pct(m.tool_accuracy)),
        ("Right tool and args", pct(m.arg_accuracy)),
        ("Steps per task", f"{m.mean_steps:.2f}"),
        ("Latency p50 / p95", f"{m.latency_p50_s:.3f}s / {m.latency_p95_s:.3f}s"),
        ("Tokens per task in / out", f"{m.mean_input_tokens:.0f} / {m.mean_output_tokens:.0f}"),
        ("Cost per task", usd(m.cost_per_task_usd)),
        ("By category", ", ".join(f"{c} {pct(v)}" for c, v in m.success_by_category.items())),
        ("Failures", ", ".join(f"{c} {n}" for c, n in m.failures.items()) or "none"),
    ]
    note = (
        '<p class="note">Fake LLM: the agent is scripted, so these results check the harness, not a model. '
        "Token counts are estimates.</p>"
        if status.llm == "fake"
        else ""
    )

    by_id = {t.id: t for t in tasks}
    trace_by_id = {t.task_id: t for t in traces}
    rows = []
    for s in m.per_task:
        trace = trace_by_id[s.task_id]
        calls = "".join(
            f"<li><code>{e(c.tool)}({e(str(c.args))})</code>"
            + (f' <span class="err">error: {e(c.error)}</span>' if c.error else "")
            + "</li>"
            for c in trace.tool_calls
        )
        detail = (
            f"<p><b>Prompt:</b> {e(by_id[s.task_id].prompt)}</p>"
            f"<p><b>Answer:</b> {e(trace.answer or trace.error or '(none)')}</p>"
            f"<ol>{calls}</ol>"
        )
        rows.append(
            f'<tr class="{"ok" if s.success else "fail"}">'
            f"<td><details><summary>{e(s.task_id)}</summary>{detail}</details></td>"
            f"<td>{e(s.category)}</td><td>{'yes' if s.success else 'no'}</td>"
            f"<td>{e(s.failure.value) if s.failure else ''}</td>"
            f"<td>{s.right_args}/{s.expected_calls}</td><td>{s.steps}</td>"
            f"<td>{s.wall_time_s:.3f}s</td><td>{s.input_tokens}/{s.output_tokens}</td></tr>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run {e(status.name)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 70rem; padding: 0 1rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
th, td {{ border-bottom: 1px solid #ddd; padding: .35rem .6rem; text-align: left; vertical-align: top; }}
tr.fail td:first-child {{ border-left: 4px solid #c0392b; }}
tr.ok td:first-child {{ border-left: 4px solid #27ae60; }}
.note {{ background: #fff4d6; padding: .5rem .8rem; }}
.err {{ color: #c0392b; }}
code {{ font-size: .85em; }}
</style></head><body>
<h1>Run {e(status.name)}</h1>
<p>Model <code>{e(status.model)}</code> ({e(status.llm)}), {status.tasks} tasks, started {e(status.created_at)},
id <code>{e(status.id)}</code></p>
{note}
<table>{"".join(f"<tr><th>{k}</th><td>{e(v)}</td></tr>" for k, v in overview)}</table>
<h2>Tasks</h2>
<table><tr><th>Task</th><th>Category</th><th>Success</th><th>Failure</th><th>Right calls</th>
<th>Steps</th><th>Latency</th><th>Tokens in/out</th></tr>
{"".join(rows)}
</table></body></html>
"""
