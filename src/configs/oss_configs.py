from src.configs.models import Model, RunConfig, Step, StepRevisionPool

oss_model = Model.openrouter_gpt_oss_120b

oss_config = RunConfig(
    final_follow_model=oss_model,
    final_follow_times=5,
    max_concurrent_tasks=40,
    steps=[
        Step(
            instruction_model=oss_model,
            follow_model=oss_model,
            times=10,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=oss_model,
            follow_model=oss_model,
            times=20,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevisionPool(
            top_scores_used=5,
            instruction_model=oss_model,
            follow_model=oss_model,
            times=5,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)
