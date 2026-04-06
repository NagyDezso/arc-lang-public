# ARC Lang Solver

ARC Lang is an asynchronous pipeline for tackling Abstraction and Reasoning Corpus (ARC) puzzles with language models.  It iteratively prompts models to write instructions, tests those instructions on the training grids, revises the best ideas, and finally applies the strongest instructions to produce two candidate outputs for each test grid.

## How the system works

- **Dataset loading** – `src/run.py` parses ARC challenge JSON files (see `data/arc-prize-20XX/`).  Challenges are processed in batches with a monitored semaphore so multiple tasks can run in parallel without exceeding API limits.
- **Instruction generation** – For each `Step` in a `RunConfig`, `get_instruction_scores` prompts an LLM (defined by `step.instruction_model`) with the training grids via `src/main.py`.  Each response is scored by leave-one-out cross validation: the instructions are applied to every training example using `output_grid_from_instructions`, which is another LLM call that follows the instructions.
- **Scoring** – `score_instructions_on_challenge` records per-example results, calculates a simple cell-wise similarity score, writes attempts to Postgres if `NEON_DSN` is set, and keeps the top instructions in memory.
- **Revision and pooling** – `StepRevision` asks the model to repair its own instructions using a rich feedback prompt that highlights wrong outputs.  `StepRevisionPool` synthesizes a new plan from the best previous instructions and their scores.  Both feed back into the scoring loop.
- **Final predictions** – `return_answer` replays the strongest instructions with `final_follow_model` to generate multiple outputs per test grid.  The system picks up to two diverse guesses per grid and writes them under `results/<run-timestamp>/attempts/`.  If ground-truth solutions are supplied, `evaluate_solutions` computes accuracy; otherwise the guesses are ready for competition submission.

## Repository layout

- `src/run.py` – async entry point and orchestration of the entire solve loop.
- `src/main.py` – prompt builders for instruction creation, revision, and grid execution.
- `src/configs/` – ready-to-use `RunConfig` presets (`grok_config_prod`, `mini_config`, `oss_config`, etc.).
- `src/llms/` – provider wrappers and structured output helpers (`get_next_structure`).
- `src/models.py` – Pydantic models for ARC challenges, helper utilities, and visualization support.
- `src/async_utils/semaphore_monitor.py` – concurrency guard that logs semaphore saturation.
- `results/` – per-run folders `results/<YYYY-MM-DD_HH-MM-SS>/` containing `arc.log`, an `attempts/` subdirectory (per-task JSON plus the aggregate submission file), and anything else you attach to that run.
- `data/` – ARC datasets (training, evaluation, and ground-truth solutions where available).

## Requirements

- Python 3.12+ (project targets 3.12 via Ruff configuration).
- Access tokens for the model providers you intend to use.  The default config uses xAI Grok, but other presets rely on OpenAI, Anthropic, Gemini, DeepSeek, or OpenRouter.
- `MAX_CONCURRENCY` environment variable – required; sets the global API semaphore inside `src/llms/structured.py`.

Install dependencies with either `uv` or `pip`:

```bash
uv sync
```

## Environment configuration

Environment variables are loaded automatically from a `.env` file courtesy of `python-dotenv` inside `src/logging_config.py`.  A typical configuration looks like:

```dotenv
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
OPENROUTER_API_KEY=...
XAI_API_KEY=key1,key2   # multiple keys allowed for grok
MAX_CONCURRENCY=20

LOGFIRE_API_KEY=...     # optional remote logging
LOCAL_LOGS_ONLY=1       # skip sending logs upstream
LOG_LEVEL=INFO
# LOG_FILE=/absolute/path/arc.log   # optional; overrides results/<timestamp>/arc.log

NEON_DSN=postgresql://...  # optional; enables result persistence
USE_TASK_ID=0              # set to 1 to send true task ids to logs/LLMs
VIZ=0                      # set to 1 to open matplotlib previews during scoring
LOG_GRIDS=0                # set to 1 to log mismatched grids verbosely
```

Only the API keys for the models you select and `MAX_CONCURRENCY` are strictly required.  If a variable is missing the related feature simply falls back (for example, no Postgres writes when `NEON_DSN` is unset).

## Running a solve

The main entry point is `src/run.py`.  It currently wires up the 2025 evaluation challenges and a concrete `RunConfig` inside `run()` (adjust `limit`, `task_ids`, and the imported preset there for your workload).

```bash
python src/run.py
```

What happens:

1. A run id is generated (`logging_config.generate_run_id`) and stored in log context.
2. The evaluation challenges JSON is loaded, optionally filtered by `limit`, `offset`, or `task_ids`.
3. Each challenge is solved asynchronously via `solve_challenges`, constrained by `config.max_concurrent_tasks`.
4. A new directory `results/<timestamp>/` is created.  Local logs go to `results/<timestamp>/arc.log` (unless `LOG_FILE` is set).  Per-task guesses are written to `results/<timestamp>/attempts/<task_id>.json`, and the rolling aggregate submission file lives in the same `attempts/` folder.
5. If you pass a solutions JSON and call `evaluate_solutions` afterward (as `run()` does), final accuracy is printed; otherwise submit the aggregate attempt JSON from that run’s `attempts/` folder.

To run on more tasks, adjust the call inside `run()`—for example set `limit=None` to sweep the full evaluation set or swap in another config:

```python
aggregate_path = await run_from_json(
    challenges_path=challenges_path,
    truth_solutions_path=solutions_path,  # required: ground-truth JSON for scoring in this flow
    config=mini_config,                   # e.g. faster preset
    limit=None,
    offset=0,
)
# aggregate_path is results/<timestamp>/attempts/arc-agi_<split>_attempts.json
```

Omit `attempts_path` and `temp_attempts_dir` to use the default layout under `results/<timestamp>/attempts/`.  Pass both together if you want a custom location.  Optionally set `results_run_dir` to fix the timestamp folder.  `run_from_json` returns the aggregate attempts file path so you can pass it straight to `evaluate_solutions`.

You can also call `run_from_json` from your own script with custom paths, a bespoke `RunConfig`, or overrides for `results_run_dir` / attempts paths.

## RunConfig primer

`src/configs/models.py` defines three step types:

- `Step` – generate `times` instruction candidates with `instruction_model`, score them by following the instructions with `follow_model`.
- `StepRevision` – take the top `top_scores_used` candidates, ask the LLM to revise each `times_per_top_score` times, then rescore using the revision’s `follow_model`.
- `StepRevisionPool` – build a synthesis prompt that shows multiple instruction sets, including per-example scores, and request a brand-new instruction `times` times.

A `RunConfig` bundles a sequence of these steps plus:

- `final_follow_model` and `final_follow_times` for the last pass when we produce answers for the hidden test grids.
- `max_concurrent_tasks` to bound how many challenges are solved at once.

By editing or creating a `RunConfig` you control which providers are called, whether images or diff notations are included, and how aggressively the system revises its plans.  The presets in `src/configs/` illustrate lightweight (`mini_config`), production Grok (`grok_config_prod`), GPT-5 (`gpt_config_prod`), and fully open-source (`oss_config`) strategies.

## Outputs and persistence

- Run folder: each `run_from_json` invocation uses `results/<YYYY-MM-DD_HH-MM-SS>/` (or `results_run_dir` if you pass one).  The console prints `Results directory: ...` at start.
- Attempt JSON: `results/<timestamp>/attempts/arc-agi_<split>_attempts.json` contains both guesses per task in the format expected by ARC competition submissions (derived from the challenges filename by swapping `_challenges` for `_attempts`).
- Per-task files: `results/<timestamp>/attempts/<task_id>.json` holds that task’s guesses so you can inspect intermediate results while the run is in flight.
- Optional database writes: when `NEON_DSN` is present, each `InstructionsScore` and final `Guess` is inserted into Postgres for analysis.
- Local logs: during a solve, `results/<timestamp>/arc.log` receives structured spans and key-value metadata (via `configure_local_log_path`), unless `LOG_FILE` points elsewhere.  Short-lived imports that never run `run_from_json` still default to `logs/arc.log`.  Remote Logfire emission is enabled unless `LOCAL_LOGS_ONLY=1`.

## Debugging and visualization tips

- Set `VIZ=1` to open matplotlib comparisons whenever a training grid prediction differs from the target (requires a display or X forwarding).
- Toggle `LOG_GRIDS=1` to dump expected vs. actual grids in logs.
- `generate_grid_diff` (exposed in `src/run.py`) can be reused in notebooks to produce ASCII diffs of grid pairs.
- Use `results/<timestamp>/attempts/<task_id>.json` to inspect what the model produced for each test grid.

## Development

- Ruff and mypy configs live in `pyproject.toml`.  Run `uv run ruff check src` or `uv run mypy src` to lint/type-check.
- The project is asyncio-first; if you extend it, prefer async functions and reuse `MonitoredSemaphore` to avoid overload.
- `src/llms/structured.py` centralizes provider-specific settings (retries, pricing metadata, structured output formats).  Extend this module if you add a new model family.

## Troubleshooting

- Missing `MAX_CONCURRENCY` will raise a `KeyError` on import—define it before running.
- Authentication errors typically surface inside `get_next_structure`; double-check API keys and provider quotas.
- If you see repeated `retry_failed` logs, the provider may be rate-limiting you—lower `max_concurrent_tasks` or `MAX_CONCURRENCY`.
- When running on headless servers, keep `VIZ=0` to avoid matplotlib backend errors.

With the README and code as reference, you can adapt ARC Lang to new model providers, tweak scoring heuristics, or plug in alternative instruction synthesis strategies.
