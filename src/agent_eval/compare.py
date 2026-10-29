"""Compares agent configs (model x prompt) on the same tasks and checks for regressions.

A comparison file names system prompts and the runs to make:

    prompts:
      terse: You answer questions about an outdoor-gear shop ...
    runs:
      - {name: mini-default, llm: openai, model: gpt-4o-mini}
      - {name: mini-terse, llm: openai, model: gpt-4o-mini, prompt: terse}

The prompt `default` is the runner's SYSTEM_PROMPT unless the file defines it. Every run
gets the same tasks, its trace is saved as <out_dir>/<name>.jsonl, and the scored runs are
written as a markdown table and as JSON. The JSON can be kept as a baseline: a later
comparison fails the regression check if a run's success rate dropped by more than a
threshold on the tasks both runs share.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from agent_eval.llm import LLM, FakeLLM, OpenAIClient
from agent_eval.metrics import Price, RunMetrics, score_run
from agent_eval.runner import SYSTEM_PROMPT, LLMAgent, run
from agent_eval.tasks import Task


class RunConfig(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9._-]+$")  # also the trace file name
    llm: Literal["fake", "openai"] = "fake"
    model: str | None = None  # openai: the model to request; fake: ignored
    prompt: str = "default"
    give_up_every: int = Field(0, ge=0)  # fake only: every n-th task answers without tools

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        if self.llm == "openai" and not self.model:
            raise ValueError(f"run {self.name}: an openai run needs a model")
        if self.llm != "fake" and self.give_up_every:
            raise ValueError(f"run {self.name}: give_up_every only applies to the fake LLM")
        return self

    @property
    def model_name(self) -> str:
        """Key into the price table."""
        return self.model if self.llm == "openai" else "fake"


class CompareConfig(BaseModel):
    prompts: dict[str, str] = Field(default_factory=dict)
    runs: list[RunConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> CompareConfig:
        self.prompts.setdefault("default", SYSTEM_PROMPT)
        names = [r.name for r in self.runs]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate run names: {', '.join(duplicates)}")
        unknown = sorted({r.prompt for r in self.runs} - self.prompts.keys())
        if unknown:
            raise ValueError(f"unknown prompts: {', '.join(unknown)}")
        return self


def load_config(path: Path) -> CompareConfig:
    return CompareConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def build_agent(cfg: RunConfig, prompts: dict[str, str], tasks: list[Task]) -> LLMAgent:
    """The agent for one run. Its name is the run name, so traces of different runs stay apart."""
    if cfg.llm == "fake":
        llm: LLM = FakeLLM.from_tasks(tasks, give_up_every=cfg.give_up_every)
    else:
        llm = OpenAIClient.from_env(model=cfg.model)
    return LLMAgent(llm, system_prompt=prompts[cfg.prompt], name=cfg.name)


class RunResult(BaseModel):
    name: str
    llm: str
    model: str
    prompt: str
    metrics: RunMetrics


class Comparison(BaseModel):
    tasks: int
    runs: list[RunResult]


def compare(
    config: CompareConfig, tasks: list[Task], prices: dict[str, Price], out_dir: Path | None = None
) -> Comparison:
    """Run every config on `tasks` and score it. Traces go to `out_dir` if given."""
    results = []
    for cfg in config.runs:
        agent = build_agent(cfg, config.prompts, tasks)
        traces = run(agent, tasks, out=None if out_dir is None else out_dir / f"{cfg.name}.jsonl")
        price = prices.get(cfg.model_name)
        metrics = score_run(tasks, traces, {cfg.name: price} if price else None)
        results.append(
            RunResult(name=cfg.name, llm=cfg.llm, model=cfg.model_name, prompt=cfg.prompt, metrics=metrics)
        )
    return Comparison(tasks=len(tasks), runs=results)


def markdown_table(comparison: Comparison) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.0%}"

    def usd(x: float | None) -> str:
        return "n/a" if x is None else f"${x:.5f}"

    lines = [
        (
            "| Run | Model | Prompt | Success | Right tool | Right tool and args | Steps | "
            "Latency p50/p95 | Tokens in/out | Cost per task | Failures |"
        ),
        (
            "|-----|-------|--------|--------:|-----------:|--------------------:|------:|"
            "----------------:|--------------:|--------------:|----------|"
        ),
    ]
    for r in comparison.runs:
        m = r.metrics
        failures = ", ".join(f"{c} {n}" for c, n in sorted(m.failures.items(), key=lambda x: (-x[1], x[0])))
        lines.append(
            f"| {r.name} | {r.model} | {r.prompt} | {m.succeeded}/{m.tasks} ({pct(m.success_rate)}) "
            f"| {pct(m.tool_accuracy)} | {pct(m.arg_accuracy)} | {m.mean_steps:.2f} "
            f"| {m.latency_p50_s:.3f}s/{m.latency_p95_s:.3f}s "
            f"| {m.mean_input_tokens:.0f}/{m.mean_output_tokens:.0f} | {usd(m.cost_per_task_usd)} "
            f"| {failures or '-'} |"
        )
    return "\n".join(lines)


def write_results(comparison: Comparison, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / "results.md", out_dir / "results.json"
    md.write_text(markdown_table(comparison) + "\n", encoding="utf-8")
    js.write_text(comparison.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return md, js


def load_comparison(path: Path) -> Comparison:
    return Comparison.model_validate(json.loads(path.read_text(encoding="utf-8")))


class RunCheck(BaseModel):
    name: str
    shared_tasks: int
    baseline_rate: float
    current_rate: float
    newly_failed: list[str]
    regressed: bool

    def line(self) -> str:
        verdict = "REGRESSION" if self.regressed else "ok"
        delta = self.current_rate - self.baseline_rate
        text = (
            f"{verdict} {self.name}: success {self.baseline_rate:.1%} -> {self.current_rate:.1%} "
            f"({delta:+.1%}) on {self.shared_tasks} shared tasks"
        )
        if self.newly_failed:
            text += "; newly failed: " + ", ".join(self.newly_failed)
        return text


def check_regression(current: Comparison, baseline: Comparison, threshold: float) -> list[RunCheck]:
    """Compare each run with the baseline run of the same name, on the tasks both ran.

    A run regressed if its success rate dropped by more than `threshold` (0.05 = 5 points).
    Runs missing from either side are not compared.
    """
    base = {r.name: r for r in baseline.runs}
    checks = []
    for r in current.runs:
        if r.name not in base:
            continue
        now = {s.task_id: s.success for s in r.metrics.per_task}
        before = {s.task_id: s.success for s in base[r.name].metrics.per_task}
        shared = sorted(now.keys() & before.keys())
        if not shared:
            continue
        before_rate = sum(before[t] for t in shared) / len(shared)
        now_rate = sum(now[t] for t in shared) / len(shared)
        checks.append(
            RunCheck(
                name=r.name,
                shared_tasks=len(shared),
                baseline_rate=before_rate,
                current_rate=now_rate,
                newly_failed=[t for t in shared if before[t] and not now[t]],
                # rounded so a drop of exactly the threshold is not a regression
                regressed=round(before_rate - now_rate, 9) > threshold,
            )
        )
    return checks
