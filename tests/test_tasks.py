from pathlib import Path

import pytest

from agent_eval.tasks import Category, Task, load_tasks
from agent_eval.validate import check_task

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"
TASKS = load_tasks(TASKS_DIR)


def test_benchmark_size_and_mix():
    assert len(TASKS) == 30
    categories = {t.category for t in TASKS}
    assert categories == set(Category)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_gold_calls_ground_the_answer(task):
    assert check_task(task) == []


def test_ungrounded_answer_is_caught():
    task = Task.model_validate(
        {
            "id": "bad-001",
            "category": "sql",
            "difficulty": "easy",
            "prompt": "How many customers are from Germany?",
            "expected_calls": [
                {"tool": "sql_query", "args": {"query": "SELECT COUNT(*) FROM customers WHERE country = 'Germany'"}}
            ],
            "expected_answer": {"type": "numeric", "value": 4},
        }
    )
    assert check_task(task) == ["answer 4.0 not found in tool outputs"]


def test_unknown_tool_is_caught():
    task = Task.model_validate(
        {
            "id": "bad-002",
            "category": "docs",
            "difficulty": "easy",
            "prompt": "x",
            "expected_calls": [{"tool": "web_search", "args": {"query": "x"}}],
            "expected_answer": {"type": "contains", "values": ["x"]},
        }
    )
    assert "unknown tool 'web_search'" in check_task(task)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"expected_calls": []}, "at least one expected call"),
        ({"category": "refusal"}, "refusal"),
        ({"expected_answer": {"type": "contains", "values": []}}, "at least one entry"),
        ({"expected_answer": {"type": "numeric"}}, "need `value`"),
    ],
)
def test_schema_rejects_malformed_tasks(overrides, message):
    raw = {
        "id": "bad-003",
        "category": "sql",
        "difficulty": "easy",
        "prompt": "x",
        "expected_calls": [{"tool": "sql_query", "args": {"query": "SELECT 1"}}],
        "expected_answer": {"type": "numeric", "value": 1},
    } | overrides
    with pytest.raises(ValueError, match=message):
        Task.model_validate(raw)


def test_duplicate_ids_are_rejected(tmp_path):
    task = (TASKS_DIR / "refusal.yaml").read_text(encoding="utf-8")
    (tmp_path / "a.yaml").write_text(task, encoding="utf-8")
    (tmp_path / "b.yaml").write_text(task, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task ids"):
        load_tasks(tmp_path)
