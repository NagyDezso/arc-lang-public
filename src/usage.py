"""Persisted per-task token usage / cost for a results run.

Each solved task writes ``<run_dir>/usage/<task_id>.json`` holding the token
usage broken down by model. submit.py and the end-of-run summary read these
files, so cost survives resumes (already-solved tasks keep their usage file)
without re-parsing logs.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from pydantic import BaseModel

from src.llms.models import TokenUsage

USAGE_DIRNAME = "usage"


def usage_dir_for_run(run_dir: Path) -> Path:
    return run_dir / USAGE_DIRNAME


class TaskUsage(BaseModel):
    """Token usage for one task, split by the model that produced it."""

    task_id: str
    usage_by_llm: dict[str, TokenUsage] = {}
    max_single_call_total_tokens: int = 0

    def total_usage(self) -> TokenUsage:
        total = TokenUsage()
        for usage in self.usage_by_llm.values():
            total += usage
        return total

    def cost(self) -> float:
        return round(
            sum(usage.cost(llm) for llm, usage in self.usage_by_llm.items()), 2
        )


def write_task_usage(run_dir: Path, task_usage: TaskUsage) -> None:
    usage_dir = usage_dir_for_run(run_dir)
    usage_dir.mkdir(parents=True, exist_ok=True)
    (usage_dir / f"{task_usage.task_id}.json").write_text(
        task_usage.model_dump_json(), encoding="utf-8"
    )


def load_task_usages(run_dir: Path) -> list[TaskUsage]:
    usage_dir = usage_dir_for_run(run_dir)
    out: list[TaskUsage] = []
    if not usage_dir.is_dir():
        return out
    for path in sorted(usage_dir.glob("*.json")):
        try:
            out.append(TaskUsage.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


class RunUsageSummary(BaseModel):
    usage_by_llm: dict[str, TokenUsage]
    task_costs: dict[str, float]
    total_cost: float

    @property
    def total_usage(self) -> TokenUsage:
        total = TokenUsage()
        for usage in self.usage_by_llm.values():
            total += usage
        return total


def summarize_run_usage(run_dir: Path) -> RunUsageSummary | None:
    tasks = load_task_usages(run_dir)
    if not tasks:
        return None

    usage_by_llm: dict[str, TokenUsage] = {}
    task_costs: dict[str, float] = {}
    for task in tasks:
        task_costs[task.task_id] = task.cost()
        for llm, usage in task.usage_by_llm.items():
            usage_by_llm.setdefault(llm, TokenUsage())
            usage_by_llm[llm] += usage

    total_cost = round(
        sum(usage.cost(llm) for llm, usage in usage_by_llm.items()), 2
    )
    return RunUsageSummary(
        usage_by_llm=usage_by_llm, task_costs=task_costs, total_cost=total_cost
    )


def format_run_usage(summary: RunUsageSummary) -> str:
    lines = ["=== Cost ==="]
    for llm in sorted(summary.usage_by_llm):
        usage = summary.usage_by_llm[llm]
        lines.append(
            f"{llm}: ${usage.cost(llm):.2f} "
            f"(in={usage.input_tokens:,} out={usage.output_tokens:,} "
            f"reasoning={usage.reasoning_tokens:,} cached={usage.cached_tokens:,})"
        )
    lines.append(f"Total cost: ${summary.total_cost:.2f}")
    costs = list(summary.task_costs.values())
    if costs:
        lines.append(
            f"Per task ({len(costs)} tasks): "
            f"mean=${statistics.mean(costs):.2f} "
            f"median=${statistics.median(costs):.2f}"
        )
    return "\n".join(lines)


def print_run_usage(run_dir: Path) -> None:
    summary = summarize_run_usage(run_dir)
    if summary is None:
        return
    print(f"\n{format_run_usage(summary)}")
