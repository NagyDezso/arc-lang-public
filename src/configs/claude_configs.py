"""Run configs that route every LLM call through the Claude Code CLI (``claude``).

Each call runs in its own temp cwd with session persistence disabled (see
:mod:`src.llms.claude_code`), so calls parallelize freely — pick
``max_concurrent_tasks`` based on what the subscription quota and CPU can take.
Auth is whatever the local ``claude`` CLI is logged into (a Claude Pro/Max
subscription), so no API tokens are billed.
"""

from src.configs.models import RunConfig, Step, StepRevisionPool

llm = "claudecode/sonnet"

claude_sonnet_config = RunConfig(
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
