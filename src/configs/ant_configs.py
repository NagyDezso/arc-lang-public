from src.configs.models import Model, RunConfig, Step, StepRevisionPool

sonnet_4_5_config_prod = RunConfig(
    final_follow_model=Model.sonnet_4_5,
    final_follow_times=5,
    max_concurrent_tasks=120,
    steps=[
        Step(
            instruction_model=Model.sonnet_4_5,
            follow_model=Model.sonnet_4_5,
            times=5,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=Model.sonnet_4_5,
            follow_model=Model.sonnet_4_5,
            times=5,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=Model.sonnet_4_5,
            follow_model=Model.sonnet_4_5,
            times=20,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        # StepRevision(
        #     top_scores_used=5,
        #     instruction_model=Model.sonnet_4_5,
        #     follow_model=Model.sonnet_4_5,
        #     times_per_top_score=1,
        #     timeout_secs=300,
        #     include_base64=False,
        #     use_diffs=True,
        # ),
        StepRevisionPool(
            top_scores_used=5,
            instruction_model=Model.sonnet_4_5,
            follow_model=Model.sonnet_4_5,
            times=5,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)
