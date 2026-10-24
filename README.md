# Agent Eval Platform

A benchmark harness that tests whether tool-using AI agents work reliably, not just in a demo.

Agents get a realistic environment (a fictional outdoor-gear shop with a SQL database and policy
documents) and a set of tasks with known correct tool calls and answers. The harness will measure
task success, tool-call accuracy, latency, token cost and failure types, and compare models and prompts.

## Status

- [x] Task format and 30 benchmark tasks, with a validator that proves every task is answerable
- [ ] Runner that executes an agent on every task and records each step
- [ ] Metrics: success rate, tool-call accuracy, latency p50/p95, cost per task
- [ ] Failure classes: wrong tool, bad arguments, loops and timeouts, ungrounded answers
- [ ] Model and prompt comparison with regression checks
- [ ] FastAPI service and HTML report
- [ ] Docker Compose and CI

## The environment

| Tool | What it does |
|------|--------------|
| `sql_query` | Read-only SELECT on the shop database: `customers`, `products`, `orders`, `order_items` ([data/shop.sql](data/shop.sql)) |
| `search_docs` | Keyword search over the policy documents ([data/docs/](data/docs/)) |
| `read_doc` | Full text of one document |

Each task gets a fresh in-memory copy, so runs are isolated and repeatable. Writes are rejected at
the database level (`PRAGMA query_only`), not only by prompt.

## Tasks

| Category | Count | Tests whether the agent can |
|----------|------:|-----------------------------|
| `sql` | 12 | write correct queries, from simple counts to multi-table joins |
| `docs` | 8 | find the right document and pull out the exact policy |
| `multi` | 6 | combine a database fact with a policy (e.g. order total plus the shipping rule) |
| `refusal` | 4 | decline writes and admit when the data does not contain the answer |

A task looks like this ([tasks/](tasks/)):

```yaml
- id: multi-004
  category: multi
  difficulty: medium
  prompt: Anna Schmidt wants to change the delivery address of order 15. Is that still possible?
  expected_calls:
    - tool: sql_query
      args: {query: "SELECT status FROM orders WHERE id = 15"}
    - tool: search_docs
      args: {query: change delivery address}
    - tool: read_doc
      args: {doc_id: support}
  expected_answer: {type: contains, values: [shipped, [cannot, not possible, no longer possible]]}
```

Answer types: `contains` (all values must appear; a nested list lists accepted spellings),
`numeric` (within a tolerance), and `refusal`.

## The benchmark is checked, too

A benchmark with wrong gold answers produces wrong conclusions. `validate` replays each task's
expected tool calls in a fresh environment and fails if a call errors, if `search_docs` does not
surface the document the task expects, or if the expected answer is not in the tool outputs.

```bash
pip install -e ".[dev]"
python -m agent_eval validate   # 30 tasks: 8 docs, 6 multi, 4 refusal, 12 sql / all tasks valid
pytest
```
