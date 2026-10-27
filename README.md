# Agent Eval Platform

A benchmark harness that tests whether tool-using AI agents work reliably, not just in a demo.

Agents get a realistic environment (a fictional outdoor-gear shop with a SQL database and policy
documents) and a set of tasks with known correct tool calls and answers. The harness will measure
task success, tool-call accuracy, latency, token cost and failure types, and compare models and prompts.

## Status

- [x] Task format and 30 benchmark tasks, with a validator that proves every task is answerable
- [x] Runner that executes an agent on every task and records each step
- [x] Metrics: success rate, tool-call accuracy, latency p50/p95, cost per task
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

## Running an agent

The runner gives every task a fresh environment, lets the agent call tools until it answers
without a tool call, and stops it at the task's `max_steps`. Each step (one model call plus the
tool calls it asked for) is recorded: the messages, every tool call with its arguments and result
or error, input and output tokens, and wall time. Traces are written to JSONL, one task per line.

```bash
python -m agent_eval run                      # fake LLM, writes runs/fake.jsonl
python -m agent_eval run --task multi-004     # a single task

# any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, ...)
export AGENT_EVAL_MODEL=gpt-4o-mini
export AGENT_EVAL_BASE_URL=https://api.openai.com/v1   # the default
export AGENT_EVAL_API_KEY=sk-...
python -m agent_eval run --llm openai          # writes runs/gpt-4o-mini.jsonl
```

The default `FakeLLM` needs no API key. It replays a fixed script per task, so tests and CI are
free and give the same trace every time. Its default script is a perfect agent (the gold tool calls,
then the gold answer), which checks the plumbing, not a model. Its token counts are estimates
(4 characters per token), not real usage.

| Backend | Tasks | Answered | Steps |
|---------|------:|---------:|------:|
| `FakeLLM` (scripted, not a real model) | 30 | 30 | 76 |

## Metrics

`run` scores the trace as soon as it finishes; `score` re-scores a saved one.

```bash
python -m agent_eval score runs/fake.jsonl --json runs/fake.metrics.json
```

| Metric | How it is measured |
|--------|--------------------|
| Task success | The final answer is checked with the task's `ExpectedAnswer` (`contains`, `numeric`, `refusal`). Hitting `max_steps` is a failure, and so is a refusal after the agent tried to write. |
| Tool-call accuracy | Share of expected calls the agent made with the right tool, and with the right tool and arguments. Arguments are compared by effect, not text: `sql_query` must return the same rows (in any order), `search_docs` must surface the document the task reads next, `read_doc` must name the same document. |
| Steps | Model calls per task. |
| Latency | Wall time per task, p50 and p95 across tasks. |
| Tokens and cost | Input and output tokens per task, priced with [config/prices.yaml](config/prices.yaml) (USD per million tokens, keyed by model name). Models missing from the table get no cost. |

Measured with the scripted `FakeLLM` (a perfect agent that replays the gold calls, not a real
model, so this only checks that the scoring agrees with the benchmark; its tokens are estimates):

| Backend | Success | Right tool | Right tool and args | Steps per task | Tokens per task in/out |
|---------|--------:|-----------:|--------------------:|---------------:|-----------------------:|
| `FakeLLM` (scripted, not a real model) | 30/30 | 100% | 100% | 2.53 | 1208/98 (estimated) |

Real model results: run `python -m agent_eval run --llm openai` with the environment variables
above and add the numbers here.

| Model | Success | Right tool | Right tool and args | Latency p50/p95 | Cost per task |
|-------|--------:|-----------:|--------------------:|----------------:|--------------:|
| | | | | | |
