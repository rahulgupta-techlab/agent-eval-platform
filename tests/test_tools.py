import pytest

from agent_eval.tools import MAX_ROWS, Environment, ToolError


@pytest.fixture
def env():
    return Environment()


def test_sql_returns_columns_and_rows(env):
    result = env.call("sql_query", {"query": "SELECT name FROM products WHERE id = 1"})
    assert result == {"columns": ["name"], "rows": [["Trail Runner Shoes"]], "truncated": False}


@pytest.mark.parametrize(
    "query",
    ["DELETE FROM orders", "UPDATE products SET price = 1", "DROP TABLE customers"],
)
def test_database_is_read_only(env, query):
    with pytest.raises(ToolError):
        env.call("sql_query", {"query": query})
    assert env.call("sql_query", {"query": "SELECT COUNT(*) FROM orders"})["rows"] == [[20]]


def test_large_results_are_truncated(env):
    query = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 100) SELECT i FROM n"
    result = env.call("sql_query", {"query": query})
    assert len(result["rows"]) == MAX_ROWS
    assert result["truncated"]


def test_search_ranks_relevant_doc_first(env):
    hits = env.call("search_docs", {"query": "tent poles warranty"})
    assert hits[0]["doc_id"] == "warranty"
    assert "poles" in hits[0]["snippet"]


def test_search_with_no_match_is_empty(env):
    assert env.call("search_docs", {"query": "xylophone"}) == []


def test_read_unknown_doc_lists_available(env):
    with pytest.raises(ToolError, match="Available: loyalty"):
        env.call("read_doc", {"doc_id": "privacy"})


def test_bad_tool_calls_raise_tool_error(env):
    with pytest.raises(ToolError, match="unknown tool"):
        env.call("send_email", {})
    with pytest.raises(ToolError, match="bad arguments"):
        env.call("read_doc", {"id": "returns"})


def test_schemas_are_openai_style(env):
    names = [t.schema()["function"]["name"] for t in env.tools.values()]
    assert names == ["sql_query", "search_docs", "read_doc"]
