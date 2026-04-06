from src.configs.models import Model, RunConfig, Step, StepRevisionPool

model = Model.gpt_5_pro

gpt_config_prod = RunConfig(
    final_follow_model=model,
    final_follow_times=5,
    max_concurrent_tasks=120,
    steps=[
        Step(
            instruction_model=model,
            follow_model=model,
            times=5,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=model,
            follow_model=model,
            times=5,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=model,
            follow_model=model,
            times=20,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
        # StepRevision(
        #     top_scores_used=5,
        #     instruction_model=model,
        #     follow_model=model,
        #     times_per_top_score=1,
        #     timeout_secs=10_800,
        #     include_base64=False,
        #     use_diffs=True,
        # ),
        StepRevisionPool(
            top_scores_used=5,
            instruction_model=model,
            follow_model=model,
            times=5,
            timeout_secs=10_800,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)
