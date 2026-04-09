from pydantic import BaseModel, field_validator

from src.llms.models import parse_llm


class StepBase(BaseModel):
    instruction_llm: str
    follow_llm: str

    include_base64: bool
    use_diffs: bool

    timeout_secs: int

    @field_validator("instruction_llm", "follow_llm")
    @classmethod
    def _validate_llm(cls, v: str) -> str:
        parse_llm(v)
        return v


class Step(StepBase):
    times: int


class StepRevision(StepBase):
    top_scores_used: int
    times_per_top_score: int


class StepRevisionPool(StepBase):
    top_scores_used: int
    times: int


class RunConfig(BaseModel):
    final_follow_llm: str
    final_follow_times: int
    max_concurrent_tasks: int

    steps: list[Step | StepRevision | StepRevisionPool]

    @field_validator("final_follow_llm")
    @classmethod
    def _validate_final_llm(cls, v: str) -> str:
        parse_llm(v)
        return v
