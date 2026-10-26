import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agent_eval.__main__ import main
from agent_eval.llm import FakeLLM, LLMError, OpenAIClient, Turn
from agent_eval.runner import LLMAgent, load_traces, run, run_task
from agent_eval.tasks import load_tasks
from agent_eval.tools import Environment

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"
TASKS = load_tasks(TASKS_DIR)
SQL_001 = next(t for t in TASKS if t.id == "sql-001")


def scripted(*turns):
    return LLMAgent(FakeLLM({SQL_001.prompt: list(turns)}))


def test_perfect_fake_agent_answers_every_task():
    traces = run(LLMAgent(FakeLLM.from_tasks(TASKS)), TASKS)
    assert [t.task_id for t in traces] == [t.id for t in TASKS]
    for task, trace in zip(TASKS, traces):
        assert trace.status == "answered"
        assert len(trace.steps) == len(task.expected_calls) + 1
        assert all(c.error is None for s in trace.steps for c in s.tool_calls)
        assert trace.input_tokens > 0 and trace.output_tokens > 0


def test_step_records_messages_tool_calls_and_tokens():
    query = "SELECT COUNT(*) FROM customers WHERE country = 'Germany'"
    trace = run_task(scripted(Turn(calls=[("sql_query", {"query": query})]), Turn(content="3")), SQL_001)
    first, last = trace.steps
    assert [m["role"] for m in first.messages] == ["system", "user", "assistant", "tool"]
    assert first.tool_calls[0].tool == "sql_query"
    assert first.tool_calls[0].args == {"query": query}
    assert first.tool_calls[0].result["rows"] == [[3]]
    assert [m["role"] for m in last.messages] == ["assistant"]
    assert trace.answer == "3"
    assert first.wall_time_s >= 0 and first.input_tokens > 0


def test_max_steps_is_enforced():
    loop = Turn(calls=[("sql_query", {"query": "SELECT 1"})])
    trace = run_task(scripted(*[loop] * 20), SQL_001)
    assert trace.status == "max_steps"
    assert len(trace.steps) == SQL_001.max_steps
    assert trace.answer is None


def test_tool_errors_are_recorded_and_shown_to_the_agent():
    trace = run_task(
        scripted(
            Turn(calls=[("sql_query", {"query": "SELECT * FROM refunds"})]),
            Turn(calls=[("send_email", {"to": "x"})]),
            Turn(content="I don't know."),
        ),
        SQL_001,
    )
    sql_error, unknown_tool = (s.tool_calls[0] for s in trace.steps[:2])
    assert "no such table: refunds" in sql_error.error
    assert "unknown tool" in unknown_tool.error
    assert trace.steps[0].messages[-1]["content"].startswith("Error: SQL error")
    assert trace.status == "answered"


def test_invalid_json_arguments_are_a_tool_error():
    class BadArgs(FakeLLM):
        def complete(self, messages, tools):
            response = super().complete(messages, tools)
            for call in response.tool_calls:
                call.arguments = '{"query": '
            return response

    llm = BadArgs({SQL_001.prompt: [Turn(calls=[("sql_query", {})]), Turn(content="?")]})
    record = run_task(LLMAgent(llm), SQL_001).steps[0].tool_calls[0]
    assert record.args is None
    assert "not valid JSON" in record.error


def test_every_task_gets_a_fresh_environment():
    created = []

    def factory():
        created.append(Environment())
        return created[-1]

    run(LLMAgent(FakeLLM.from_tasks(TASKS[:3])), TASKS[:3], env_factory=factory)
    assert len(created) == 3 and len({id(e) for e in created}) == 3


def test_llm_errors_end_the_task():
    class Down:
        name = "down"

        def complete(self, messages, tools):
            raise LLMError("connection refused")

    trace = run_task(LLMAgent(Down()), SQL_001)
    assert trace.status == "llm_error"
    assert trace.error == "connection refused"
    assert trace.steps == []


def test_traces_round_trip_through_jsonl(tmp_path):
    out = tmp_path / "run.jsonl"
    traces = run(LLMAgent(FakeLLM.from_tasks(TASKS)), TASKS, out=out)
    assert len(out.read_text(encoding="utf-8").splitlines()) == len(TASKS)
    assert load_traces(out) == traces


def test_cli_run_writes_trace(tmp_path, capsys):
    out = tmp_path / "fake.jsonl"
    assert main(["run", str(TASKS_DIR), "--task", "sql-001", "--task", "docs-001", "--out", str(out)]) == 0
    assert "2 tasks: 2 answered" in capsys.readouterr().out
    assert [t.task_id for t in load_traces(out)] == ["docs-001", "sql-001"]


def test_cli_rejects_unknown_task_id(tmp_path):
    assert main(["run", str(TASKS_DIR), "--task", "nope", "--out", str(tmp_path / "x.jsonl")]) == 2


@pytest.fixture
def chat_server(monkeypatch):
    """Local stand-in for an OpenAI-compatible endpoint. Records requests, replays `replies`."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    state = {"requests": [], "replies": [], "status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            payload = json.dumps(state["replies"].pop(0)).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    server.shutdown()


def chat_reply(content=None, tool_calls=None, usage=(120, 15)):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }


def test_openai_client_runs_a_task(chat_server, monkeypatch):
    query = "SELECT COUNT(*) FROM customers WHERE country = 'Germany'"
    function = {"name": "sql_query", "arguments": json.dumps({"query": query})}
    call = {"id": "c1", "type": "function", "function": function}
    chat_server["replies"] = [chat_reply(tool_calls=[call]), chat_reply(content="3 customers.", usage=(200, 5))]
    monkeypatch.setenv("AGENT_EVAL_MODEL", "test-model")
    monkeypatch.setenv("AGENT_EVAL_BASE_URL", chat_server["url"])
    monkeypatch.setenv("AGENT_EVAL_API_KEY", "sk-test")

    trace = run_task(LLMAgent(OpenAIClient.from_env()), SQL_001)

    assert trace.status == "answered" and trace.answer == "3 customers."
    assert trace.agent == "test-model"
    assert (trace.input_tokens, trace.output_tokens) == (320, 20)
    assert trace.steps[0].tool_calls[0].result["rows"] == [[3]]
    first, second = chat_server["requests"]
    assert first["path"] == "/v1/chat/completions"
    assert first["auth"] == "Bearer sk-test"
    assert first["body"]["model"] == "test-model"
    assert [t["function"]["name"] for t in first["body"]["tools"]] == ["sql_query", "search_docs", "read_doc"]
    tool_result = json.dumps({"columns": ["COUNT(*)"], "rows": [[3]], "truncated": False})
    assert second["body"]["messages"][-1] == {"role": "tool", "tool_call_id": "c1", "content": tool_result}


def test_openai_client_http_error_becomes_llm_error(chat_server):
    chat_server["status"] = 500
    chat_server["replies"] = [{"error": "overloaded"}]
    trace = run_task(LLMAgent(OpenAIClient("m", base_url=chat_server["url"])), SQL_001)
    assert trace.status == "llm_error"
    assert trace.error.startswith("HTTP 500")


def test_openai_client_needs_a_model(monkeypatch):
    monkeypatch.delenv("AGENT_EVAL_MODEL", raising=False)
    with pytest.raises(LLMError, match="AGENT_EVAL_MODEL"):
        OpenAIClient.from_env()
