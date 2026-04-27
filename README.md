# Stem Agent

An evolutionary meta-agent that autonomously evolves a specialised Deep Research agent from a minimal "stem cell" configuration — without hand-engineering the target architecture.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        StemAgent (orchestrator)              │
│                                                             │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────┐  │
│  │ PopulationMgr│   │ Evolution Ops  │   │  LineageDB   │  │
│  │  (in-memory  │   │ seed / llm /   │   │  (SQLite)    │  │
│  │  + DB-backed)│   │ random / xover │   │              │  │
│  └──────┬───────┘   └───────┬────────┘   └──────────────┘  │
│         │ AgentSpec         │ mutate                        │
│         ▼                   ▼                               │
│  ┌────────────────────────────────────┐                     │
│  │       ResearchRunner (async)       │                     │
│  │  plan_and_solve | react loop       │                     │
│  │  token budget + retry + tool cache │                     │
│  └──────────────────┬─────────────────┘                     │
│                     │ RunResult                             │
│         ┌───────────▼───────────┐                           │
│         │  Verifiable Scorer    │   (zero LLM — 100% det.) │
│         │  + optional LLM Judge │   (--qualitative only)   │
│         └───────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Deterministic verifiable scorer | LLM judges introduce noise; exact-match / tolerance scoring provides a stable fitness signal |
| AsyncIO evaluation loop | 8× speed improvement vs. sequential; `asyncio.Semaphore` prevents API rate-limit hammering |
| Architectural mutations | LLM mutator can change `tools`, `planning_strategy`, `stop_condition` — not just prompt text |
| Single-writer SQLite lineage DB | Prevents database-locked errors under concurrent async writes |
| Subprocess REPL sandbox | Eliminates raw `exec()` security hole; hard timeout + restricted env |
| Graduation threshold | Evolution stops at `best_score >= 0.85`; prevents over-engineering |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key"

# Or create a .env file:
OPENAI_API_KEY=your-api-key
```

### 3. Run

```bash
# Full run (15 generations, verifiable tasks)
python main.py

# Quick smoke test (3 generations, fast tasks)
python main.py --quick

# Full benchmark on held-out test set after evolution
python main.py --benchmark

# Qualitative run: LLM judge on 5 open-ended tasks after evolution
python main.py --qualitative

# Combine flags
python main.py --benchmark --qualitative
```

## Evaluation

The primary fitness signal is fully **deterministic** — no LLM involved during evolution:

- **factoid**: regex number extraction + string matching
- **numeric**: relative tolerance window (configurable per task)
- **set_comparison**: Jaccard similarity against required answer set
- **list_comparison**: intersection/union, order-insensitive

The `--qualitative` flag re-runs the evolved agent on 5 open-ended research questions and scores them with a separate LLM judge, providing the human-readable quality check.

## Datasets

| File | Purpose | Tasks |
|---|---|---|
| `tasks/verifiable_tasks.jsonl` | Training set (evolution signal) | 30 |
| `tasks/regression.jsonl` | Canary check for new mutations | 10 |
| `tasks/test_verifiable.jsonl` | **Held-out** benchmark (never seen during evolution) | 10 |
| `tasks/train.jsonl` | Open-ended qualitative tasks | 10 |

## Configuration (`config.yaml`)

Key parameters:

```yaml
evolution:
  generations:          15    # max generations to run
  population_size:       5    # agents kept per generation
  patience:              3    # stop if no improvement for N generations
  random_mutation_rate: 0.5   # probability of random mutation per parent
  concurrency:           8    # parallel (spec, task) evaluations
  task_timeout_s:       120   # seconds per single task run
  graduation_threshold: 0.85  # stop early if best score reaches this

runtime:
  token_budget:         8000  # cumulative token limit per run
  sandbox_mode:     subprocess  # subprocess | docker
```

## Lineage Database

All evolution state is persisted to `logs/lineage.db` (SQLite). Tables:

- `specs` — every spec, with parent ID and mutation type
- `evaluations` — per-task scores and cause codes
- `run_traces` — full message histories for debugging
- `mutations` — parent→child mutation log
- `generations` — generation-level statistics

Evolution can be **resumed** at any time by re-running `python main.py`.

## Troubleshooting

### Evolution stalls after Generation 1
Check that `tasks/regression.jsonl` has valid tasks — the canary regression check will block all children if the tasks themselves are malformed.

### "database is locked"
Should not occur with the writer-queue design. If it does, confirm you're not running two `python main.py` processes simultaneously.

### Tool sandbox errors
The subprocess sandbox writes to `%TEMP%/stem_sandbox/`. Ensure the temp directory is writable.
