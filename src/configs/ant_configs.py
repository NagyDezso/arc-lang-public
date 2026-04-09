from src.configs.models import RunConfig, Step, StepRevisionPool

llm = "anthropic/claude-sonnet-4-5-20250929"
sonnet_4_5_config_prod = RunConfig(
    final_follow_llm=llm,
    final_follow_times=5,
    max_concurrent_tasks=120,
    steps=[
        Step(
            instruction_llm=llm,
            follow_llm=llm,
            times=5,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_llm=llm,
            follow_llm=llm,
            times=5,
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
        # StepRevision(
        #     top_scores_used=5,
        #     instruction_llm=llm,
        #     follow_llm=llm,
        #     times_per_top_score=1,
        #     timeout_secs=300,
        #     include_base64=False,
        #     use_diffs=True,
        # ),
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
