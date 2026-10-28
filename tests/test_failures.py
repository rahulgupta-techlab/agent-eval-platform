from pathlib import Path

import pytest

from agent_eval.failures import LOOP_REPEATS, FailureClass
from agent_eval.llm import FakeLLM, LLMError, Turn
from agent_eval.metrics import score_run, score_task, summary
from agent_eval.runner import LLMAgent, run, run_task
from agent_eval.tasks import load_tasks

TASKS = load_tasks(Path(__file__).resolve().parents[1] / "tasks")
BY_ID = {t.id: t for t in TASKS}
F = FailureClass


def score_of(task_id, *turns):
    task = BY_ID[task_id]
    return score_task(task, run_task(LLMAgent(FakeLLM({task.prompt: list(turns)})), task))


def gold(task_id, i=0):
    call = BY_ID[task_id].expected_calls[i]
    return Turn(calls=[(call.tool, call.args)])


def sql(query):
    return Turn(calls=[("sql_query", {"query": query})])


def answer(text):
    return Turn(content=text)


def test_successful_task_has_no_failure_class():
    s = score_of("sql-001", gold("sql-001"), answer("3"))
    assert s.success and s.failure is None


@pytest.mark.parametrize(
    "task_id, turns, expected",
    [
        # called a tool that cannot answer the question
        ("sql-001", [Turn(calls=[("search_docs", {"query": "customers Germany"})]), answer("5")], F.WRONG_TOOL),
        # called a tool that does not exist
        ("sql-001", [Turn(calls=[("lookup_customers", {"country": "Germany"})]), answer("5")], F.WRONG_TOOL),
        # right tool, query returns a different result
        ("sql-001", [sql("SELECT COUNT(*) FROM customers"), answer("12")], F.BAD_ARGUMENTS),
        # right tool for the search, but read the wrong document
        (
            "docs-001",
            [gold("docs-001"), Turn(calls=[("read_doc", {"doc_id": "warranty"})]), answer("2 years")],
            F.BAD_ARGUMENTS,
        ),
        # SQL error, never fixed
        ("sql-001", [sql("SELECT COUNT(*) FROM clients"), answer("About 5.")], F.TOOL_ERROR_NOT_RECOVERED),
        # right calls, answer contradicts the result
        ("sql-001", [gold("sql-001"), answer("There are 7 customers from Germany.")], F.UNGROUNDED_ANSWER),
        # answered from memory without calling any tool
        ("docs-001", [answer("You have 14 days.")], F.UNGROUNDED_ANSWER),
        # declined a question the data answers
        ("sql-001", [answer("I don't have access to customer data.")], F.WRONGLY_REFUSED),
        ("sql-001", [gold("sql-001"), answer("I cannot tell from this data.")], F.WRONGLY_REFUSED),
        # answered a question the data cannot answer
        ("refusal-003", [answer("Her number is 555-0100.")], F.SHOULD_HAVE_REFUSED),
        # tried to write, then declined
        (
            "refusal-001",
            [sql("DELETE FROM orders WHERE status = 'cancelled'"), answer("I cannot do that, the data is read-only.")],
            F.SHOULD_HAVE_REFUSED,
        ),
        # never answered
        ("sql-001", [sql("SELECT 1")] * 20, F.LOOP_OR_MAX_STEPS),
    ],
)
def test_failure_class(task_id, turns, expected):
    s = score_of(task_id, *turns)
    assert not s.success
    assert s.failure is expected


def test_repeated_call_is_a_loop_even_if_the_agent_answers():
    s = score_of("sql-001", *[sql("SELECT COUNT(*) FROM customers")] * LOOP_REPEATS, answer("12"))
    assert s.status == "answered" and s.failure is F.LOOP_OR_MAX_STEPS


def test_recovered_tool_error_is_not_the_failure():
    s = score_of("sql-001", sql("SELECT COUNT(*) FROM clients"), gold("sql-001"), answer("7"))
    assert s.failure is F.UNGROUNDED_ANSWER


def test_refusal_after_a_tool_error_is_blamed_on_the_error():
    s = score_of("sql-001", sql("SELECT COUNT(*) FROM clients"), answer("I cannot find that table."))
    assert s.failure is F.TOOL_ERROR_NOT_RECOVERED


def test_refusal_after_a_wrong_tool_is_blamed_on_the_tool():
    s = score_of("sql-001", Turn(calls=[("search_docs", {"query": "germany"})]), answer("I don't know."))
    assert s.failure is F.WRONG_TOOL


def test_llm_error_is_its_own_class():
    class Down:
        name = "down"

        def complete(self, messages, tools):
            raise LLMError("connection refused")

    task = BY_ID["sql-001"]
    assert score_task(task, run_task(LLMAgent(Down()), task)).failure is F.LLM_ERROR


def test_run_counts_failures_by_class():
    tasks = [BY_ID[i] for i in ("sql-001", "sql-002", "docs-001", "refusal-003")]
    llm = FakeLLM(
        {
            tasks[0].prompt: [gold("sql-001"), answer("3")],
            tasks[1].prompt: [sql("SELECT COUNT(*) FROM customers"), answer("12")],
            tasks[2].prompt: [answer("14 days.")],
            tasks[3].prompt: [answer("555-0100")],
        }
    )
    m = score_run(tasks, run(LLMAgent(llm), tasks))
    assert m.failures == {"bad_arguments": 1, "should_have_refused": 1, "ungrounded_answer": 1}
    assert sum(m.failures.values()) == m.tasks - m.succeeded
    text = summary(m)
    assert "failure classes: bad_arguments 1, should_have_refused 1, ungrounded_answer 1" in text
    assert "  docs-001: ungrounded_answer" in text


def test_perfect_fake_agent_has_no_failures():
    m = score_run(TASKS, run(LLMAgent(FakeLLM.from_tasks(TASKS)), TASKS))
    assert m.failures == {}
    assert all(s.failure is None for s in m.per_task)
