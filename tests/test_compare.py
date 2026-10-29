import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_eval.__main__ import main
from agent_eval.compare import (
    CompareConfig,
    build_agent,
    check_regression,
    compare,
    load_comparison,
    load_config,
    markdown_table,
)
from agent_eval.metrics import Price, load_prices
from agent_eval.runner import SYSTEM_PROMPT
from agent_eval.tasks import load_tasks

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "tasks"
TASKS = load_tasks(TASKS_DIR)


def config(*runs, prompts=None):
    return CompareConfig.model_validate({"prompts": prompts or {}, "runs": list(runs)})


def test_every_run_gets_the_same_tasks_and_its_own_trace(tmp_path):
    cfg = config({"name": "a", "llm": "fake"}, {"name": "b", "llm": "fake", "give_up_every": 5})
    c = compare(cfg, TASKS, load_prices(), out_dir=tmp_path)
    assert [r.name for r in c.runs] == ["a", "b"]
    assert [r.metrics.tasks for r in c.runs] == [len(TASKS)] * 2
    assert c.runs[0].metrics.succeeded == len(TASKS)
    assert c.runs[1].metrics.succeeded == len(TASKS) - len(TASKS) // 5
    assert c.runs[0].metrics.total_cost_usd == 0  # fake runs are priced as model "fake"
    for name in ("a", "b"):
        lines = (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(TASKS) and json.loads(lines[0])["agent"] == name


def test_prompt_is_used_as_the_system_message():
    cfg = config({"name": "a", "prompt": "short"}, {"name": "b"}, prompts={"short": "Be brief."})
    first, second = (build_agent(r, cfg.prompts, TASKS).start(TASKS[0])[0]["content"] for r in cfg.runs)
    assert (first, second) == ("Be brief.", SYSTEM_PROMPT)


def test_openai_run_uses_the_model_from_the_config(monkeypatch):
    monkeypatch.setenv("AGENT_EVAL_MODEL", "other-model")
    cfg = config({"name": "m", "llm": "openai", "model": "gpt-4o-mini"})
    agent = build_agent(cfg.runs[0], cfg.prompts, TASKS)
    assert (agent.name, agent.llm.model, cfg.runs[0].model_name) == ("m", "gpt-4o-mini", "gpt-4o-mini")


def test_cost_comes_from_the_model_not_the_run_name():
    cfg = config({"name": "priced", "llm": "fake"})
    c = compare(cfg, TASKS[:2], {"fake": Price(input_per_mtok=1.0, output_per_mtok=0.0)})
    m = c.runs[0].metrics
    assert m.total_cost_usd == pytest.approx(sum(s.input_tokens for s in m.per_task) / 1e6)


@pytest.mark.parametrize(
    "data, message",
    [
        ({"runs": []}, "at least 1"),
        ({"runs": [{"name": "a"}, {"name": "a"}]}, "duplicate run names: a"),
        ({"runs": [{"name": "a", "prompt": "nope"}]}, "unknown prompts: nope"),
        ({"runs": [{"name": "a", "llm": "openai"}]}, "needs a model"),
        ({"runs": [{"name": "a", "llm": "openai", "model": "m", "give_up_every": 2}]}, "only applies"),
        ({"runs": [{"name": "a/b"}]}, "pattern"),
    ],
)
def test_bad_configs_are_rejected(data, message):
    with pytest.raises(ValidationError, match=message):
        CompareConfig.model_validate(data)


def test_markdown_table_has_a_row_per_run():
    cfg = config({"name": "a"}, {"name": "b", "give_up_every": 5})
    table = markdown_table(compare(cfg, TASKS, load_prices()))
    rows = table.splitlines()
    assert len(rows) == 4
    assert rows[2].startswith("| a | fake | default | 30/30 (100%) |")
    assert "| b | fake | default | 24/30 (80%) |" in rows[3] and "ungrounded_answer 5" in rows[3]


def test_regression_is_a_drop_beyond_the_threshold():
    baseline = compare(config({"name": "a"}, {"name": "b"}), TASKS, load_prices())
    current = compare(config({"name": "a", "give_up_every": 10}, {"name": "b", "give_up_every": 30}), TASKS, {})
    checks = {c.name: c for c in check_regression(current, baseline, threshold=0.05)}
    assert checks["a"].regressed and checks["a"].current_rate == pytest.approx(0.9)
    assert checks["a"].newly_failed == ["multi-002", "sql-002", "sql-012"]
    assert not checks["b"].regressed  # one task of 30 is 3.3 points
    assert not check_regression(current, baseline, threshold=0.10)[0].regressed  # exactly at the threshold


def test_regression_compares_only_shared_tasks_and_runs():
    baseline = compare(config({"name": "a"}, {"name": "old"}), TASKS, {})
    current = compare(config({"name": "a"}, {"name": "new"}), TASKS[:5], {})
    checks = check_regression(current, baseline, threshold=0.0)
    assert [(c.name, c.shared_tasks, c.regressed) for c in checks] == [("a", 5, False)]


def test_cli_compare_writes_results_and_checks_the_baseline(tmp_path, capsys):
    cfg = tmp_path / "compare.yaml"
    cfg.write_text("runs:\n  - {name: a}\n  - {name: b, give_up_every: 5}\n", encoding="utf-8")
    first = tmp_path / "first"
    args = ["compare", str(cfg), "--tasks", str(TASKS_DIR)]
    assert main(args + ["--out-dir", str(first)]) == 0
    assert (first / "results.md").read_text(encoding="utf-8").startswith("| Run |")
    assert {r.name for r in load_comparison(first / "results.json").runs} == {"a", "b"}

    baseline = str(first / "results.json")
    assert main(args + ["--out-dir", str(tmp_path / "same"), "--baseline", baseline]) == 0
    assert "ok a: success 100.0% -> 100.0%" in capsys.readouterr().out

    cfg.write_text("runs:\n  - {name: a, give_up_every: 3}\n", encoding="utf-8")
    assert main(args + ["--out-dir", str(tmp_path / "worse"), "--baseline", baseline]) == 1
    out = capsys.readouterr().out
    assert "REGRESSION a: success 100.0% -> 66.7% (-33.3%)" in out
    assert "regression: success dropped by more than 5.0%" in out


def test_cli_compare_rejects_unknown_run_and_bad_config(tmp_path):
    assert main(["compare", str(ROOT / "config" / "compare.yaml"), "--run", "nope"]) == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("runs: []\n", encoding="utf-8")
    assert main(["compare", str(bad)]) == 2


def test_shipped_configs_load():
    assert {r.llm for r in load_config(ROOT / "config" / "compare.yaml").runs} == {"fake"}
    openai = load_config(ROOT / "config" / "compare.openai.yaml")
    assert {r.prompt for r in openai.runs} == {"default", "terse"}


def test_committed_fake_baseline_still_holds():
    current = compare(load_config(ROOT / "config" / "compare.yaml"), TASKS, load_prices())
    checks = check_regression(current, load_comparison(ROOT / "baselines" / "compare-fake.json"), 0.0)
    assert len(checks) == 2 and not any(c.regressed or c.newly_failed for c in checks)
