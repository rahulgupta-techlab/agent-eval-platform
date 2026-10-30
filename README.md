# Agent Eval Platform

A benchmark harness that tests whether tool-using AI agents work reliably, not just in a demo.

Agents get a realistic environment (a fictional outdoor-gear shop with a SQL database and policy
documents) and a set of tasks with known correct tool calls and answers. The harness will measure
task success, tool-call accuracy, latency, token cost and failure classes, and compare models and prompts.

## Status

- [x] Task format and 30 benchmark tasks, with a validator that proves every task is answerable
- [x] Runner that executes an agent on every task and records each step
- [x] Metrics: success rate, tool-call accuracy, latency p50/p95, cost per task
- [x] Failure classes: wrong tool, bad arguments, loops and timeouts, ungrounded answers
- [x] Model and prompt comparison with regression checks
- [x] FastAPI service and HTML report
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

## Failure classes

A success rate says how often an agent fails, not why. Every failed task gets exactly one primary
failure class ([failures.py](src/agent_eval/failures.py)). The checks run in this order and the
first that applies wins, so the class names the earliest point where the run went wrong rather
than every symptom that followed (a refusal after a SQL error is blamed on the error).

| Class | The task failed because |
|-------|-------------------------|
| `llm_error` | the model could not be reached (kept apart: not the agent's fault) |
| `loop_or_max_steps` | the agent hit `max_steps`, or made the same tool call (same arguments) 3 times |
| `should_have_refused` | a refusal task where the agent answered anyway, or tried to write |
| `wrongly_refused` | the agent declined a question the data answers, with no tool error or wrong call to blame |
| `tool_error_not_recovered` | a tool call failed and no later call to that tool succeeded |
| `wrong_tool` | an expected call was never made with the right tool (including tools that do not exist) |
| `bad_arguments` | the right tool, but never with the right effect (same comparison as tool-call accuracy) |
| `ungrounded_answer` | the answer is not supported by the tool outputs: no tool was called (checked right after `wrongly_refused`), or every call was right (so the outputs held the gold answer, as `validate` proves) and the answer still disagrees |

`run` and `score` print the count per class and the class of each failed task, and `--json` writes
them (`failures` per run, `failure` per task). With the scripted perfect `FakeLLM` there are no
failures, so that run only shows the plumbing; the tests in
[tests/test_failures.py](tests/test_failures.py) script one failing agent per class. Real-model
failure counts come from the same `run --llm openai` command as the metrics above.

## Comparing models and prompts

`compare` runs several configs (model x system prompt) on the same tasks, saves a trace per run
in `--out-dir`, and writes the results as `results.md` and `results.json`. Configs are YAML
([config/compare.yaml](config/compare.yaml), [config/compare.openai.yaml](config/compare.openai.yaml)):

```yaml
prompts:
  terse: You answer questions about an outdoor-gear shop. ...   # `default` is the built-in prompt
runs:
  - {name: gpt-4o-mini-default, llm: openai, model: gpt-4o-mini}
  - {name: gpt-4o-mini-terse, llm: openai, model: gpt-4o-mini, prompt: terse}
```

```bash
python -m agent_eval compare                                   # the two scripted fake runs
python -m agent_eval compare config/compare.openai.yaml --out-dir runs/openai
python -m agent_eval compare config/compare.openai.yaml --run gpt-4o-mini-terse   # one run only
```

Any `results.json` can serve as a baseline. With `--baseline`, each run is compared with the
baseline run of the same name on the tasks both ran, and the command exits with 1 if a success
rate dropped by more than `--threshold` (default 0.05, i.e. 5 points). It also lists the tasks
that passed before and fail now.

```bash
python -m agent_eval compare config/compare.openai.yaml --out-dir runs/openai \
    --baseline baselines/openai.json --threshold 0.05
```

CI runs the fake configs against [baselines/compare-fake.json](baselines/compare-fake.json) on
every push, so a change to the runner or the scoring that alters any result fails the build.
Measured with the scripted `FakeLLM` (not a real model): `fake-gives-up` is the perfect script
except that every 5th task answers at once without calling a tool. It exists to show that the
comparison and the regression check catch a worse agent; tokens are estimates.

| Run | Success | Right tool | Right tool and args | Steps | Tokens in/out | Failures |
|-----|--------:|-----------:|--------------------:|------:|--------------:|----------|
| `fake-perfect` (scripted) | 30/30 | 100% | 100% | 2.53 | 1208/98 (estimated) | - |
| `fake-gives-up` (scripted) | 24/30 | 83% | 83% | 2.27 | 1062/83 (estimated) | ungrounded_answer 5, should_have_refused 1 |

Real model results: run `python -m agent_eval compare config/compare.openai.yaml --out-dir runs/openai`
with `AGENT_EVAL_API_KEY` (and `AGENT_EVAL_BASE_URL` for a non-OpenAI endpoint) set, and paste
`runs/openai/results.md` here.

| Run | Model | Prompt | Success | Right tool | Right tool and args | Latency p50/p95 | Cost per task |
|-----|-------|--------|--------:|-----------:|--------------------:|----------------:|--------------:|
| | | | | | | | |

## HTTP service

`serve` starts a FastAPI app ([service.py](src/agent_eval/service.py)) that runs one config at a
time in the background and keeps the results in memory. Traces are saved to `runs/service/<id>.jsonl`.

| Endpoint | What it does |
|----------|--------------|
| `POST /runs` | Start a run. Body: `name`, `llm` (`fake` or `openai`), `model`, `system_prompt`, `task_ids` (default: all), `give_up_every` (fake only). Returns `202` with the run id. |
| `GET /runs` | Every run, newest first, without metrics. |
| `GET /runs/{id}` | Status (`queued`, `running`, `done`, `failed`), the error if it failed, and the full metrics once it is done. |
| `GET /runs/{id}/report` | HTML report: the metrics, then every task with its prompt, answer, tool calls and failure class. `409` until the run is done. |

```bash
python -m agent_eval serve --port 8000
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
    -d '{"name": "fake-gives-up", "give_up_every": 5}'          # {"id": "537616dd3b5a", "status": "queued", ...}
curl localhost:8000/runs/537616dd3b5a                          # status and metrics
open http://localhost:8000/runs/537616dd3b5a/report            # HTML report

# a real model: the same AGENT_EVAL_BASE_URL / AGENT_EVAL_API_KEY as above
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
    -d '{"name": "mini", "llm": "openai", "model": "gpt-4o-mini"}'
```

The tests ([tests/test_service.py](tests/test_service.py)) drive it with FastAPI's `TestClient`
and the scripted `FakeLLM`; a `fake-gives-up` run through the API gives the same 24/30 as `compare`.

