import json
from pathlib import Path

import pytest

from agent_eval.__main__ import main
from agent_eval.llm import FakeLLM, Turn
from agent_eval.metrics import (
    Price,
    answer_is_correct,
    load_prices,
    percentile,
    score_run,
    score_task,
)
from agent_eval.runner import LLMAgent, load_traces, run, run_task
from agent_eval.tasks import ExpectedAnswer, load_tasks

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"
TASKS = load_tasks(TASKS_DIR)
BY_ID = {t.id: t for t in TASKS}


def trace_of(task_id, *turns):
    task = BY_ID[task_id]
    return run_task(LLMAgent(FakeLLM({task.prompt: list(turns)})), task)


def score_of(task_id, *turns):
    return score_task(BY_ID[task_id], trace_of(task_id, *turns))


def sql(query):
    return Turn(calls=[("sql_query", {"query": query})])


@pytest.mark.parametrize(
    "answer, expected, ok",
    [
        ("The order has SHIPPED, so it cannot be changed.", {"type": "contains", "values": ["shipped", ["cannot", "not possible"]]}, True),
        ("The order has shipped.", {"type": "contains", "values": ["shipped", ["cannot", "not possible"]]}, False),
        ("Revenue was 1,234.50 EUR.", {"type": "numeric", "value": 1234.5}, True),
        ("About 162.3", {"type": "numeric", "value": 162.33}, False),
        ("162.335", {"type": "numeric", "value": 162.33}, True),
        ("I can't change prices: the database is read-only.", {"type": "refusal"}, True),
        ("Her number is 555-0100.", {"type": "refusal"}, False),
    ],
)
def test_answer_is_checked_against_expected_answer(answer, expected, ok):
    assert answer_is_correct(answer, ExpectedAnswer.model_validate(expected)) is ok


def test_sql_args_are_compared_by_result_not_text():
    s = score_of("sql-001", sql("select count(id) from customers where country='Germany'"), Turn(content="3"))
    assert (s.success, s.right_tool, s.right_args) == (True, 1, 1)


def test_sql_with_a_different_result_has_the_right_tool_but_wrong_args():
    s = score_of("sql-001", sql("SELECT COUNT(*) FROM customers"), Turn(content="3"))
    assert (s.right_tool, s.right_args) == (1, 0)


def test_sql_error_is_not_right_args():
    s = score_of("sql-001", sql("SELECT COUNT(*) FROM clients"), Turn(content="3"))
    assert (s.right_tool, s.right_args) == (1, 0)


def test_wrong_tool_scores_nothing():
    s = score_of("sql-001", Turn(calls=[("search_docs", {"query": "customers Germany"})]), Turn(content="3"))
    assert (s.right_tool, s.right_args, s.calls_made) == (0, 0, 1)


def test_search_args_are_right_if_the_needed_document_is_found():
    # the expected search's top hit is `loyalty`, but the task needs `shipping`
    good = Turn(calls=[("search_docs", {"query": "express delivery days"})])
    bad = Turn(calls=[("search_docs", {"query": "loyalty points"})])
    read = Turn(calls=[("read_doc", {"doc_id": "shipping"})])
    assert score_of("docs-004", good, read, Turn(content="?")).right_args == 2
    assert score_of("docs-004", bad, read, Turn(content="?")).right_args == 1


def test_each_actual_call_matches_one_expected_call():
    task = BY_ID["multi-002"]
    query = task.expected_calls[0].args["query"]
    s = score_of("multi-002", Turn(calls=[("sql_query", {"query": query})]), Turn(content="?"))
    assert s.expected_calls == len(task.expected_calls) > 1
    assert (s.right_tool, s.right_args) == (1, 1)


def test_unfinished_task_fails():
    s = score_of("sql-001", *[sql("SELECT 1")] * 20)
    assert s.status == "max_steps" and not s.success and s.steps == BY_ID["sql-001"].max_steps


def test_refusal_fails_if_a_write_was_attempted():
    delete = sql("DELETE FROM orders WHERE status = 'cancelled'")
    refuse = Turn(content="I cannot do that, the data is read-only.")
    assert score_of("refusal-001", refuse).success
    assert not score_of("refusal-001", delete, refuse).success


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5.0], 95) == 5.0
    assert percentile(list(range(101)), 95) == 95
    assert percentile([], 50) == 0.0


def test_cost_uses_the_price_table():
    trace = trace_of("sql-001", sql("SELECT 1"), Turn(content="3"))
    price = Price(input_per_mtok=1.0, output_per_mtok=4.0)
    s = score_task(BY_ID["sql-001"], trace, price)
    assert s.cost_usd == pytest.approx((trace.input_tokens + 4 * trace.output_tokens) / 1e6)
    assert score_task(BY_ID["sql-001"], trace).cost_usd is None


def test_price_table_loads():
    prices = load_prices()
    assert prices["fake"].input_per_mtok == 0


def test_perfect_fake_agent_scores_full_marks():
    m = score_run(TASKS, run(LLMAgent(FakeLLM.from_tasks(TASKS)), TASKS), load_prices())
    assert m.succeeded == m.tasks == len(TASKS)
    assert m.tool_accuracy == m.arg_accuracy == 1.0
    assert set(m.success_by_category) == {"docs", "multi", "refusal", "sql"}
    assert m.mean_steps == sum(len(t.expected_calls) + 1 for t in TASKS) / len(TASKS)
    assert m.total_cost_usd == 0
    assert m.latency_p50_s <= m.latency_p95_s


def test_run_metrics_aggregate_failures():
    tasks = [BY_ID["sql-001"], BY_ID["sql-002"]]
    llm = FakeLLM(
        {
            tasks[0].prompt: [Turn(calls=[("sql_query", tasks[0].expected_calls[0].args)]), Turn(content="3")],
            tasks[1].prompt: [Turn(calls=[("read_doc", {"doc_id": "returns"})]), Turn(content="I don't know.")],
        }
    )
    m = score_run(tasks, run(LLMAgent(llm), tasks))
    assert (m.succeeded, m.success_rate) == (1, 0.5)
    assert m.tool_accuracy == m.arg_accuracy == 0.5
    assert m.cost_per_task_usd is None


def test_score_run_rejects_mixed_agents_and_unknown_tasks():
    a = trace_of("sql-001", Turn(content="3"))
    b = a.model_copy(update={"agent": "other"})
    with pytest.raises(ValueError, match="more than one agent"):
        score_run(TASKS, [a, b])
    with pytest.raises(ValueError, match="unknown tasks"):
        score_run(TASKS[:1], [a])


def test_cli_score_writes_json(tmp_path, capsys):
    trace = tmp_path / "fake.jsonl"
    assert main(["run", str(TASKS_DIR), "--task", "sql-001", "--out", str(trace)]) == 0
    out = tmp_path / "metrics.json"
    assert main(["score", str(trace), "--tasks", str(TASKS_DIR), "--json", str(out)]) == 0
    assert "success: 1/1" in capsys.readouterr().out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["success_rate"] == 1.0 and data["per_task"][0]["task_id"] == "sql-001"
    assert load_traces(trace)[0].task_id == "sql-001"
