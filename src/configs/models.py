from pydantic import BaseModel

from src.llms.models import Model


class StepBase(BaseModel):
    instruction_model: Model
    follow_model: Model

    include_base64: bool
    use_diffs: bool

    timeout_secs: int


class Step(StepBase):
    times: int


class StepRevision(StepBase):
    top_scores_used: int
    times_per_top_score: int


class StepRevisionPool(StepBase):
    top_scores_used: int
    times: int


class RunConfig(BaseModel):
    final_follow_model: Model
    final_follow_times: int
    max_concurrent_tasks: int

    steps: list[Step | StepRevision | StepRevisionPool]
