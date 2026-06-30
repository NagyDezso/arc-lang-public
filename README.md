# ARC Lang Solver

ARC Lang is an asynchronous pipeline for tackling Abstraction and Reasoning Corpus (ARC) puzzles with language models.  It iteratively prompts models to write instructions, tests those instructions on the training grids, revises the best ideas, and finally applies the strongest instructions to produce two candidate outputs for each test grid.

## How the system works

- **Dataset loading** – `src/run.py` parses ARC challenge JSON files (see `data/arc-prize-20XX/`).  Challenges are processed in batches with a monitored semaphore so multiple tasks can run in parallel without exceeding API limits.
- **Instruction generation** – For each `Step` in a `RunConfig`, `get_instruction_scores` prompts an LLM (defined by `step.instruction_llm`) with the training grids via `src/main.py`.  Each response is scored by leave-one-out cross validation: the instructions are applied to every training example using `output_grid_from_instructions`, which is another LLM call that follows the instructions.
- **Scoring** – `score_instructions_on_challenge` records per-example results, calculates a simple cell-wise similarity score, writes attempts to Postgres if `NEON_DSN` is set, and keeps the top instructions in memory.
- **Revision and pooling** – `StepRevision` asks the model to repair its own instructions using a rich feedback prompt that highlights wrong outputs.  `StepRevisionPool` synthesizes a new plan from the best previous instructions and their scores.  Both feed back into the scoring loop.
- **Final predictions** – `return_answer` replays the strongest instructions with `final_follow_llm` to generate multiple outputs per test grid.  The system picks up to two diverse guesses per grid and writes them under `results/<run-timestamp>/attempts/`.  If ground-truth solutions are supplied, `evaluate_solutions` computes accuracy; otherwise the guesses are ready for competition submission.

## Repository layout

- `src/run.py` – async entry point, CLI, and orchestration of the entire solve loop.
- `src/main.py` – prompt builders for instruction creation, revision, and grid execution.
- `src/configs/` – ready-to-use `RunConfig` presets (`gpt52_config_prod`, `mini_config`, `oss_config`, `agy_flash_config`, `claude_sonnet_config`, `gemini3pro_config_prod`, etc.).
- `src/llms/` – provider wrappers and structured output helpers (`get_next_structure`); `clients.py` holds the shared SDK clients, `agy.py` and `claude_code.py` are the headless-CLI providers.
- `src/submit.py` – standalone CLI that scores a saved run’s attempts against ARC solutions and audits `agy` transcripts.
- `src/usage.py` – per-task token/cost accounting and the end-of-run cost summary.
- `src/notify.py` – optional ntfy.sh push notifications.
- `src/models.py` – Pydantic models for ARC challenges, helper utilities, and visualization support.
- `src/async_utils/semaphore_monitor.py` – concurrency guard that logs semaphore saturation.
- `results/` – per-run folders `results/<YYYY-MM-DD_HH-MM-SS>/` containing `arc.log`, an `attempts/` subdirectory (per-task JSON plus the aggregate submission file), a `usage/` subdirectory (per-task token usage), and—when the `agy` provider is used—an `agy_transcripts/` subdirectory.
- `data/` – ARC datasets (training, evaluation, and ground-truth solutions where available).

## Requirements

- Python 3.12+ (project targets 3.12 via Ruff configuration).
- Access tokens for the model providers you intend to use.  Every step references a model through a `provider/model_id` handle, and the supported providers are `openai`, `anthropic`, `gemini`, `xai`, `deepseek`, `openrouter`, `groq`, `kilo`, `copilot`, `lmstudio`, `gateway` (Pydantic AI gateway / Google Vertex), plus the two headless-CLI providers `agy` (Google Antigravity) and `claudecode` (Claude Code).  The default config (`gpt52_config_prod`) uses OpenAI GPT-5.2.
- `MAX_CONCURRENCY` environment variable – required; sets the global API semaphore inside `src/llms/structured.py`.
- The `agy` and `claudecode` providers additionally require their CLIs (`agy`, `claude`) to be installed and reachable on `PATH`.

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
GROQ_API_KEY=...
KILO_API_KEY=...
COPILOT_API_KEY=...                  # optional; defaults to "copilot" against http://localhost:4141/v1
LMSTUDIO_API_KEY=...                 # optional; defaults to "lm-studio" against http://127.0.0.1:4444/v1
PYDANTIC_AI_GATEWAY_API_KEY=...      # gateway provider (Google Vertex via Pydantic AI)
XAI_API_KEY=key1,key2                # multiple keys allowed for grok; one is chosen at random
MAX_CONCURRENCY=20

# Headless CLI providers (no API key billed; they reuse a local CLI login)
CLAUDE_CODE_OAUTH_TOKEN=...                 # optional; else falls back to ~/.claude/.credentials.json
CLAUDE_CODE_SESSION_LIMIT_THRESHOLD=0.70    # pause claudecode calls once the 5h session window is this full (0 disables)
ANTIGRAVITY_OAUTH_REFRESH_TOKEN=...         # optional; else uses the agy CLI's existing login
# AGY_DATA_DIR=~/.gemini/antigravity-cli    # optional; override the agy data dir cloned per call
# AGY_TRANSCRIPT_DIR=...                     # optional; defaults to results/<timestamp>/agy_transcripts

LOGFIRE_API_KEY=...     # optional remote logging
LOCAL_LOGS_ONLY=1       # skip sending logs upstream
LOG_LEVEL=INFO
# LOG_FILE=/absolute/path/arc.log   # optional; overrides results/<timestamp>/arc.log

NEON_DSN=postgresql://...  # optional; enables result persistence
NTFY_TOPIC=...             # optional; push run notifications via ntfy.sh
USE_TASK_ID=0              # set to 1 to send true task ids to logs/LLMs
VIZ=0                      # set to 1 to open matplotlib previews during scoring
LOG_GRIDS=0                # set to 1 to log mismatched grids verbosely
```

Only the API keys for the models you select and `MAX_CONCURRENCY` are strictly required.  If a variable is missing the related feature simply falls back (for example, no Postgres writes when `NEON_DSN` is unset, and no push notifications when `NTFY_TOPIC` is unset).

## Running a solve

The main entry point is `src/run.py`.  It wires up the 2025 evaluation challenges and the default preset (`gpt52_config_prod`) inside `run()`, sweeping the full evaluation set (`limit=None`).  A small CLI controls the rest:

```bash
python src/run.py                                  # full eval set, default config
python src/run.py --model openrouter/qwen/qwen3.5  # override every step's model handle
python src/run.py --task <task_id> --task <id2>    # run only specific task ids (repeat or comma-separate)
python src/run.py --resume results/2025-01-01_12-00-00  # continue an existing run, skipping finished tasks
```

`--model/-m` takes a `provider/model_id` handle and rewrites `instruction_llm`, `follow_llm`, and `final_follow_llm` on every step of the default config via `with_llm`.

What happens:

1. A run id is generated (`logging_config.generate_run_id`) and stored in log context.
2. The evaluation challenges JSON is loaded, optionally filtered by `limit`, `offset`, or `--task` ids.
3. Each challenge is solved asynchronously via `solve_challenges`, constrained by `config.max_concurrent_tasks`.
4. A new directory `results/<timestamp>/` is created (or the `--resume` folder is reused).  Local logs go to `results/<timestamp>/arc.log` (unless `LOG_FILE` is set).  Per-task guesses are written to `results/<timestamp>/attempts/<task_id>.json`, and the rolling aggregate submission file lives in the same `attempts/` folder.
5. If you pass a solutions JSON and call `evaluate_solutions` afterward (as `run()` does), final accuracy and the run cost summary are printed; otherwise submit the aggregate attempt JSON from that run’s `attempts/` folder.

To script custom workloads, call `run_from_json` directly—for example set `limit=None` to sweep the full evaluation set or swap in another config:

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

`src/configs/models.py` defines three step types.  Each step names its models with `provider/model_id` handles (`instruction_llm`, `follow_llm`) and carries `timeout_secs`, plus `include_base64` (attach grid images) and `use_diffs` (include diff notation) flags:

- `Step` – generate `times` instruction candidates with `instruction_llm`, score them by following the instructions with `follow_llm`.
- `StepRevision` – take the top `top_scores_used` candidates, ask the LLM to revise each `times_per_top_score` times, then rescore using the revision’s `follow_llm`.
- `StepRevisionPool` – build a synthesis prompt that shows multiple instruction sets (the top `top_scores_used`), including per-example scores, and request a brand-new instruction `times` times.

A `RunConfig` bundles a sequence of these steps plus:

- `final_follow_llm` and `final_follow_times` for the last pass when we produce answers for the hidden test grids.
- `max_concurrent_tasks` to bound how many challenges are solved at once.

By editing or creating a `RunConfig` you control which providers are called, whether images or diff notations are included, and how aggressively the system revises its plans.  The presets in `src/configs/` illustrate the default GPT-5.2 production config (`gpt52_config_prod`), a fast Groq-backed config (`mini_config`), fully open-source via OpenRouter (`oss_config`), GPT-5 Pro (`gpt_config_prod`), Gemini 3 Pro across direct/gateway/OpenRouter routes (`gemini3pro_*`), and the headless-CLI configs `agy_flash_config` (Antigravity) and `claude_sonnet_config` (Claude Code).

## Headless CLI providers

Two providers do not call an HTTP API at all.  Instead they drive a locally installed CLI in one-shot `--print` mode and reuse whatever subscription that CLI is already logged into, so no API key is billed.  Both flatten the structured-message format to a single text prompt, disable the CLI's built-in tools, and run each call in an isolated temporary home/working directory so parallel calls do not collide.

- **`agy` (Google Antigravity CLI)** – `src/llms/agy.py`.  Model handles `agy/gemini-3.5-flash`, `agy/gemini-3.5-flash-high`, `agy/gemini-3.1-pro`, `agy/gemini-3.1-pro-high`, and `agy/claude-sonnet-4-6` map to the CLI's human-readable display names in `~/.gemini/antigravity-cli/settings.json`.  Because `agy --print` only emits plain text (no `--model` flag, no structured-output mode), the adapter pins the model through settings, clamps every prompt with an explicit "do not call any tools" instruction, parses the reply as JSON matching the requested schema, and reconstructs billed tokens from a statusline hook.  It retries on quota exhaustion by parsing the `Resets in Xh Ym Zs` window from the CLI's log, and saves each conversation transcript under `AGY_TRANSCRIPT_DIR` so the run can later be audited for tool use or network access.  Authentication uses the CLI's existing login or `ANTIGRAVITY_OAUTH_REFRESH_TOKEN`.
- **`claudecode` (Claude Code CLI)** – `src/llms/claude_code.py`.  Model handles `claudecode/sonnet`, `claudecode/opus`, and `claudecode/haiku` run `claude --print --output-format json --json-schema <schema> --tools ""` and read the schema-validated `structured_output` field.  Authentication uses the local CLI login (`~/.claude/.credentials.json`) or `CLAUDE_CODE_OAUTH_TOKEN`; the project's `ANTHROPIC_API_KEY` is deliberately stripped from the child environment so calls stay on the subscription rather than the paid API.  Before each call the adapter checks the rolling 5-hour session-usage window via the OAuth usage endpoint and pauses until it resets once utilization passes `CLAUDE_CODE_SESSION_LIMIT_THRESHOLD` (default `0.70`, set `0` to disable).

## Outputs and persistence

- Run folder: each `run_from_json` invocation uses `results/<YYYY-MM-DD_HH-MM-SS>/` (or `results_run_dir` if you pass one).  The console prints `Results directory: ...` at start.
- Attempt JSON: `results/<timestamp>/attempts/arc-agi_<split>_attempts.json` contains both guesses per task in the format expected by ARC competition submissions (derived from the challenges filename by swapping `_challenges` for `_attempts`).
- Per-task files: `results/<timestamp>/attempts/<task_id>.json` holds that task’s guesses so you can inspect intermediate results while the run is in flight.
- Token usage and cost: per-task token counts are written to `results/<timestamp>/usage/<task_id>.json`; at the end of a run (and when `evaluate_solutions` runs) a `=== Cost ===` summary is printed, aggregating cost per model handle and per task using the pricing table in `src/llms/models.py`.
- agy transcripts: when the `agy` provider is used, every conversation transcript is copied to `results/<timestamp>/agy_transcripts/`.  `evaluate_solutions` scans them via `check_transcripts` and reports any tool use, network access, or secret/env probing.
- Optional database writes: when `NEON_DSN` is present, each `InstructionsScore` and final `Guess` is inserted into Postgres for analysis.
- Local logs: during a solve, `results/<timestamp>/arc.log` receives structured spans and key-value metadata (via `configure_local_log_path`), unless `LOG_FILE` points elsewhere.  Short-lived imports that never run `run_from_json` still default to `logs/arc.log`.  Remote Logfire emission is enabled unless `LOCAL_LOGS_ONLY=1`.

### Scoring a saved run

`src/submit.py` re-scores a finished run directory against ARC ground-truth solutions without re-solving anything.  It also prints the cost summary and runs the agy transcript audit:

```bash
python -m src.submit results/<timestamp>            # uses the sole attempts/*_attempts.json and the 2025 eval solutions
python -m src.submit results/<timestamp> --truth data/arc-prize-2025/arc-agi_evaluation_solutions.json --show-task-ids
```

Pass `--attempts` to point at a specific aggregate file when a run folder holds more than one.

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
