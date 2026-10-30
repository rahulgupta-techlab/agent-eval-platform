from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_eval import service
from agent_eval.llm import FakeLLM, Turn
from agent_eval.runner import LLMAgent, load_traces
from agent_eval.service import create_app
from agent_eval.tasks import load_tasks

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"
TASKS = load_tasks(TASKS_DIR)


@pytest.fixture
def client(tmp_path):
    # TestClient runs background tasks before the response returns, so a run is finished
    # by the time POST /runs gives back its id
    return TestClient(create_app(TASKS_DIR, runs_dir=tmp_path))


def start(client, **body):
    resp = client.post("/runs", json=body)
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "queued"
    return resp.json()["id"]


def test_run_on_every_task_reports_status_and_metrics(client, tmp_path):
    run_id = start(client, name="perfect")
    body = client.get(f"/runs/{run_id}").json()
    assert (body["status"], body["name"], body["llm"], body["tasks"]) == ("done", "perfect", "fake", len(TASKS))
    m = body["metrics"]
    assert (m["agent"], m["succeeded"], m["tasks"]) == ("perfect", len(TASKS), len(TASKS))
    assert m["tool_accuracy"] == m["arg_accuracy"] == 1.0
    assert len(m["per_task"]) == len(TASKS)
    assert len(load_traces(tmp_path / f"{run_id}.jsonl")) == len(TASKS)


def test_worse_agent_has_failures_in_its_metrics(client):
    run_id = start(client, name="gives-up", give_up_every=5)
    m = client.get(f"/runs/{run_id}").json()["metrics"]
    assert m["succeeded"] == len(TASKS) - len(TASKS) // 5
    assert sum(m["failures"].values()) == len(TASKS) // 5


def test_task_subset_and_custom_prompt(client, tmp_path):
    run_id = start(client, task_ids=["sql-001", "docs-001"], system_prompt="Be brief.")
    assert client.get(f"/runs/{run_id}").json()["tasks"] == 2
    traces = load_traces(tmp_path / f"{run_id}.jsonl")
    assert sorted(t.task_id for t in traces) == ["docs-001", "sql-001"]
    assert traces[0].steps[0].messages[0] == {"role": "system", "content": "Be brief."}


@pytest.mark.parametrize(
    "body",
    [
        {"task_ids": ["no-such-task"]},
        {"task_ids": []},
        {"llm": "openai"},  # an openai run needs a model
        {"llm": "openai", "model": "m", "give_up_every": 2},
        {"llm": "other"},
        {"name": "has space"},
    ],
)
def test_bad_requests_are_rejected(client, body):
    assert client.post("/runs", json=body).status_code == 422
    assert client.get("/runs").json() == []


def test_unreachable_model_is_reported_per_task(client, monkeypatch):
    monkeypatch.setenv("AGENT_EVAL_BASE_URL", "http://127.0.0.1:9")  # nothing listens on the discard port
    run_id = start(client, llm="openai", model="some-model", task_ids=["sql-001"])
    body = client.get(f"/runs/{run_id}").json()
    assert body["status"] == "done"
    assert body["metrics"]["failures"] == {"llm_error": 1}


def test_list_runs_newest_first_without_metrics(client):
    first, second = start(client, name="a", task_ids=["sql-001"]), start(client, name="b", task_ids=["sql-001"])
    listed = client.get("/runs").json()
    assert [r["id"] for r in listed] == [second, first]
    assert all(r["metrics"] is None for r in listed)


def test_unknown_run_is_404(client):
    assert client.get("/runs/nope").status_code == 404
    assert client.get("/runs/nope/report").status_code == 404


def test_report_is_html_with_every_task(client):
    run_id = start(client, name="gives-up", give_up_every=5)
    resp = client.get(f"/runs/{run_id}/report")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    page = resp.text
    assert page.startswith("<!doctype html>")
    assert "Fake LLM" in page
    assert all(f"<summary>{t.id}</summary>" in page for t in TASKS)
    assert page.count('<tr class="fail">') == len(TASKS) // 5
    assert "ungrounded_answer" in page


def test_report_escapes_model_output(client, monkeypatch):
    prompt = next(t.prompt for t in TASKS if t.id == "sql-001")
    script = FakeLLM({prompt: [Turn(content="<script>alert(1)</script>")]})
    monkeypatch.setattr(service, "build_agent", lambda cfg, prompts, tasks: LLMAgent(script, name=cfg.name))
    run_id = start(client, task_ids=["sql-001"])
    page = client.get(f"/runs/{run_id}/report").text
    assert "<script>" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_failed_run_reports_the_error_and_has_no_report(client, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(service, "run", broken)
    run_id = start(client, task_ids=["sql-001"])
    body = client.get(f"/runs/{run_id}").json()
    assert (body["status"], body["error"], body["metrics"]) == ("failed", "RuntimeError: disk full", None)
    assert client.get(f"/runs/{run_id}/report").status_code == 409
