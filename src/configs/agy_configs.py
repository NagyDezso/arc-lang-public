"""Run configs that route every LLM call through the Antigravity CLI (``agy``).

Each agy call runs in its own isolated ``$HOME`` (see :mod:`src.llms.agy`), so
calls parallelize freely — pick ``max_concurrent_tasks`` based on what quota
and CPU can take, not on adapter constraints.
"""

from src.configs.models import RunConfig, Step, StepRevisionPool

llm = "agy/gemini-3.5-flash"

agy_flash_config = RunConfig(
    final_follow_llm=llm,
    final_follow_times=3,
    max_concurrent_tasks=4,
    steps=[
        Step(
            instruction_llm=llm,
            follow_llm=llm,
            times=3,
            timeout_secs=600,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_llm=llm,
            follow_llm=llm,
            times=3,
            timeout_secs=600,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevisionPool(
            top_scores_used=3,
            instruction_llm=llm,
            follow_llm=llm,
            times=3,
            timeout_secs=600,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)
