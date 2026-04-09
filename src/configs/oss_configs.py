from src.configs.models import RunConfig, Step, StepRevisionPool

llm = "openrouter/openai/gpt-oss-120b:free"

oss_config = RunConfig(
    final_follow_llm=llm,
    final_follow_times=5,
    max_concurrent_tasks=40,
    steps=[
        Step(
            instruction_llm=llm,
            follow_llm=llm,
            times=10,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_llm=llm,
            follow_llm=llm,
            times=20,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevisionPool(
            top_scores_used=5,
            instruction_llm=llm,
            follow_llm=llm,
            times=5,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)
