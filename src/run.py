import argparse
import asyncio
import json
import os
import traceback
import typing as T
import uuid
from datetime import datetime
from pathlib import Path

import asyncpg
from pydantic import BaseModel, TypeAdapter
from tqdm.asyncio import tqdm as tqdm_async

from src.async_utils.semaphore_monitor import MonitoredSemaphore
from src.configs.models import RunConfig, Step, StepRevision, StepRevisionPool
from src.llms.models import parse_llm
from src.llms.structured import get_next_structure
from src.log import log

# Import logging_config first to apply patches before any logfire usage
from src.logging_config import configure_local_log_path, generate_run_id, set_task_id
from src.main import (
    GRID,
    INTUITIVE_PROMPT,
    Example,
    InstructionsResponse,
    ReviseInstructionsResponse,
    contents_from_challenge,
    output_grid_from_instructions,
)
from src.models import Challenge, TestExample
from src.submit import ChallengeSolution, evaluate_solutions
from src.utils import random_str

TT = T.TypeVar("TT")


def filter_out_exceptions[TT](
    lst: T.Sequence[TT | BaseException], description: str
) -> list[TT]:
    exceptions = [instr for instr in lst if isinstance(instr, BaseException)]
    for e in exceptions:
        error_kwargs: dict[str, T.Any] = {
            "error_type": type(e).__name__,
            "error_message": str(e),
        }
        if str(e) != "no scores...":
            error_kwargs["traceback"] = "".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
        log.error(f"{description}: {type(e).__name__}", **error_kwargs)
    return [i for i in lst if not isinstance(i, BaseException)]


def challenge_ids_by_size(challenges_by_id: dict[str, Challenge]) -> list[str]:
    # sort challenges by size and return list of challenge ids
    return sorted(challenges_by_id.keys(), key=lambda k: challenges_by_id[k].size())


def with_llm(config: RunConfig, llm: str) -> RunConfig:
    """Return a copy of config with every step's LLM overridden."""
    parse_llm(llm)
    updated_steps: list[Step | StepRevision | StepRevisionPool] = []
    for step in config.steps:
        updated_steps.append(
            step.model_copy(update={"instruction_llm": llm, "follow_llm": llm})
        )

    return config.model_copy(
        update={"final_follow_llm": llm, "steps": updated_steps},
    )


class ExampleScore(BaseModel):
    example: Example
    response_output_grid: GRID
    score: float
    llm: str


REVISION_PROMPT = """
Your previous instructions were applied to the training input grids, but they did not produce the correct output grids.

Below you'll see what outputs were generated when following your instructions. Compare these incorrect outputs with the correct outputs to identify where your instructions went wrong.

Based on this feedback, provide updated instructions that correctly describe the transformation pattern. Your revised instructions must:
- Fix the specific errors you observe
- Still work correctly for ALL training examples
- Remain clear, intuitive, and general

Analyze the differences between the incorrect outputs and the correct outputs to understand the true pattern, then write improved instructions.
""".strip()


class InstructionsScore(BaseModel):
    id: str

    instructions: str
    llm: str
    example_scores: list[ExampleScore]
    score: float

    step: Step | StepRevision | StepRevisionPool

    async def save_to_db(self, c: Challenge) -> None:
        # Get database connection string from environment variable
        if "NEON_DSN" not in os.environ:
            return None

        database_url = os.environ["NEON_DSN"]

        try:
            conn = await asyncpg.connect(database_url)
        except Exception as e:
            log.warning(
                "Skipping instructions DB save (could not connect)",
                error_type=type(e).__name__,
                error_message=str(e) or repr(e),
            )
            return

        try:
            # Prepare the data
            example_scores_json = [
                {
                    "input": es.example.input,
                    "output": es.example.output,
                    "response_output_grid": es.response_output_grid,
                    "score": es.score,
                }
                for es in self.example_scores
            ]

            step_json = json.dumps(
                {
                    **self.step.model_dump(),
                    "type": type(self.step).__name__,
                }
            )

            await conn.execute(
                """
                INSERT INTO instructions (id, instructions, model, example_scores, score, task_id, task_hash, step)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                self.id,
                self.instructions,
                self.llm,
                json.dumps(example_scores_json),  # Convert to JSON string for JSONB
                self.score,
                c.task_id,
                str(hash(c)),  # Convert hash to string
                step_json,
            )
        except Exception as e:
            log.warning(
                "Skipping instructions DB save (insert failed)",
                error_type=type(e).__name__,
                error_message=str(e) or repr(e),
            )
        finally:
            await conn.close()

    async def get_revised_instructions(self, c: Challenge, step: StepRevision) -> str:
        # go thru for each example, say if it got it right
        # and then give the diff
        # and finally, give an updated instructions given this feedback

        training_example_attempts: list[GRID] | None = None
        if len(self.example_scores) == len(c.train):
            training_example_attempts = [
                sc.response_output_grid for sc in self.example_scores
            ]
        else:
            log.warn(
                "Skipping attempt feedback in revision due partial example scores",
                expected_examples=len(c.train),
                received_scores=len(self.example_scores),
            )

        messages: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": INTUITIVE_PROMPT},
                    *contents_from_challenge(
                        training_examples=c.train,
                        training_example_attempts=training_example_attempts,
                        test_inputs=c.test,
                        include_base64=step.include_base64,
                        use_diffs=step.use_diffs,
                    ),
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": self.instructions,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": REVISION_PROMPT},
                    *contents_from_challenge(
                        training_examples=c.train,
                        training_example_attempts=training_example_attempts,
                        test_inputs=c.test,
                        include_base64=step.include_base64,
                        use_diffs=step.use_diffs,
                    ),
                ],
            },
        ]

        return (
            await get_next_structure(
                structure=ReviseInstructionsResponse,
                messages=messages,
                llm=step.instruction_llm,
            )
        ).revised_instructions


def get_grid_similarity(
    ground_truth_grid: list[list[int]], sample_grid: list[list[int]]
) -> float:
    """
    Calculate similarity as the percentage of cells that match exactly.
    Returns a value between 0.0 (no matches) and 1.0 (perfect match).
    """
    if not ground_truth_grid or not sample_grid:
        return 0.0

    rows = len(ground_truth_grid)
    cols = len(ground_truth_grid[0]) if ground_truth_grid[0] else 0
    if cols == 0:
        return 0.0

    if not all(len(row) == cols for row in ground_truth_grid):
        return 0.0

    if len(sample_grid) != rows or not all(len(row) == cols for row in sample_grid):
        return 0.0

    total_cells = rows * cols
    matching_cells = 0

    for i in range(rows):
        for j in range(cols):
            if ground_truth_grid[i][j] == sample_grid[i][j]:
                matching_cells += 1

    return matching_cells / total_cells


def generate_grid_diff(
    expected_grid: list[list[int]], actual_grid: list[list[int]]
) -> str:
    """
    Generate a cell-by-cell diff notation between expected and actual grids.
    Format: ASCII grid with "|" separators where each cell shows "actual→expected" or "✓value" for matches
    """
    if not expected_grid or not actual_grid:
        return "Error: Empty grid(s)"

    # Check dimensions
    if len(expected_grid) != len(actual_grid):
        return f"Error: Grid dimension mismatch (rows: {len(expected_grid)} vs {len(actual_grid)})"

    # Calculate max width needed for proper alignment
    max_width = 0
    for expected_row, actual_row in zip(expected_grid, actual_grid, strict=False):
        if len(expected_row) != len(actual_row):
            continue
        for expected_val, actual_val in zip(expected_row, actual_row, strict=False):
            if expected_val == actual_val:
                cell_width = len(f"✓{expected_val}")
            else:
                cell_width = len(f"{actual_val}→{expected_val}")
            max_width = max(max_width, cell_width)

    # Add padding for better readability
    max_width += 2  # Space on each side of content

    diff_lines = []

    # Add top border
    num_cols = len(expected_grid[0]) if expected_grid else 0
    border = "+" + "+".join(["-" * max_width for _ in range(num_cols)]) + "+"
    diff_lines.append(border)

    for row_idx, (expected_row, actual_row) in enumerate(
        zip(expected_grid, actual_grid, strict=False)
    ):
        if len(expected_row) != len(actual_row):
            diff_lines.append(
                f"| Row {row_idx}: Error: Column count mismatch ({len(expected_row)} vs {len(actual_row)}) |"
            )
            continue

        row_cells = []
        for _, (expected_val, actual_val) in enumerate(
            zip(expected_row, actual_row, strict=False)
        ):
            if expected_val == actual_val:
                cell = f"✓{expected_val}"
            else:
                cell = f"{actual_val}→{expected_val}"
            # Center the cell content with padding
            row_cells.append(cell.center(max_width))

        diff_lines.append("|" + "|".join(row_cells) + "|")

        # Add separator between rows (except after last row)
        if row_idx < len(expected_grid) - 1:
            diff_lines.append(border)

    # Add bottom border
    diff_lines.append(border)

    return "\n".join(diff_lines)


async def get_example_score(
    instructions: str,
    training_examples: list[Example],
    test_example: Example,
    include_base64: bool,
    use_diffs: bool,
    llm: str,
) -> ExampleScore:
    from src.models import COLOR_MAP
    from src.viz import viz_many

    grid_output = await output_grid_from_instructions(
        instructions=instructions,
        training_examples=training_examples,
        test_input_grid=test_example.input,
        include_base64=include_base64,
        llm=llm,
        use_diffs=use_diffs,
        is_perfect=True,
    )
    if test_example.output == grid_output:
        log.debug(
            "Example output matches as expected",
            # example_index=test_example,
        )
        similarity_score = 1
    else:
        if os.environ.get("VIZ", "0") == "1":
            # Visualize the expected output and generated output side by side
            grids = [[test_example.output, grid_output]]
            row_colors = ["red"]  # Red border to indicate mismatch
            viz_many(grids=grids, color_map=COLOR_MAP, row_border_colors=row_colors)

        similarity_score = get_grid_similarity(
            ground_truth_grid=test_example.output, sample_grid=grid_output
        )
        log.debug(
            "Grid similarity calculated",
            similarity_score=similarity_score,
            similarity_percent=f"{similarity_score * 100:.1f}%",
        )
    example_score = ExampleScore(
        example=test_example,
        response_output_grid=grid_output,
        score=similarity_score,
        llm=llm,
    )
    return example_score


async def score_instructions_on_challenge(
    c: Challenge, instructions: str, step: Step | StepRevision | StepRevisionPool
) -> InstructionsScore:
    with log.span("score_instructions", step_type=type(step).__name__):
        futures: list = []
        for i_train in range(len(c.train)):
            temp_test = c.train[i_train]
            temp_train = c.train[0:i_train] + c.train[i_train + 1 :]
            futures.append(
                get_example_score(
                    instructions=instructions,
                    training_examples=temp_train,
                    test_example=temp_test,
                    include_base64=step.include_base64,
                    use_diffs=step.use_diffs,
                    llm=step.follow_llm,
                )
            )
        example_scores = await asyncio.gather(*futures, return_exceptions=True)
        example_scores = filter_out_exceptions(
            lst=example_scores, description="Exception in get_score_from_instructions"
        )
        score = (
            sum(s.score for s in example_scores) / len(example_scores)
            if example_scores
            else 0
        )
        log.info(
            "Instructions scored",
            score=score,
            example_count=len(example_scores),
            llm=step.instruction_llm,
        )

        instructions_score = InstructionsScore(
            id=str(uuid.uuid4()),
            instructions=instructions,
            example_scores=example_scores,
            score=score,
            llm=step.instruction_llm,
            step=step,
        )
        await instructions_score.save_to_db(c=c)
        return instructions_score


async def get_score_from_instructions(
    c: Challenge, instructions: str, step: Step | StepRevision | StepRevisionPool
) -> InstructionsScore:
    instructions_score = await score_instructions_on_challenge(
        c=c, instructions=instructions, step=step
    )
    # debug(instructions_score.score)
    return instructions_score


async def get_instructions_from_challenge(c: Challenge, step: Step) -> str:
    with log.span("get_instructions_from_challenge", step=step.model_dump()):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": INTUITIVE_PROMPT},
                    *contents_from_challenge(
                        training_examples=c.train,
                        training_example_attempts=None,
                        test_inputs=c.test,
                        include_base64=step.include_base64,
                        use_diffs=step.use_diffs,
                    ),
                ],
            }
        ]
        instructions = await get_next_structure(
            structure=InstructionsResponse,
            messages=messages,
            llm=step.instruction_llm,
        )
        return instructions.instructions


async def get_instruction_score_from_challenge(
    c: Challenge, step: Step
) -> InstructionsScore:
    instructions = await get_instructions_from_challenge(c=c, step=step)
    log.debug("Instructions generated", instructions=instructions)
    return await get_score_from_instructions(c=c, instructions=instructions, step=step)


async def get_instruction_scores(c: Challenge, step: Step) -> list[InstructionsScore]:
    futures = [
        get_instruction_score_from_challenge(c=c, step=step) for _ in range(step.times)
    ]
    results = await asyncio.gather(*futures, return_exceptions=True)
    return filter_out_exceptions(
        lst=results, description="Exception in get_multiple_scores"
    )


SYNTHESIS_PROMPT = """
Multiple expert puzzle solvers have attempted to describe the transformation pattern for these grids. Each attempt captured some aspects correctly but failed in other ways.

Below you'll find:
- Each set of proposed instructions
- The outputs produced when following those instructions
- How those outputs differ from the correct answers

Your task is to analyze why each approach partially failed and synthesize a complete, correct set of instructions.

By examining multiple flawed attempts, you can:
- Identify what each approach got right
- Understand what each approach missed
- Recognize common misconceptions about the pattern
- Build comprehensive instructions that avoid all these pitfalls

Study the patterns of success and failure across all attempts, then write instructions that correctly describe the complete transformation rule that works for ALL training examples.

Your final instructions should be clear, intuitive, and capture the true underlying pattern.
    """.strip()


async def get_pooling_instruction_from_scores(
    c: Challenge, scores: list[InstructionsScore], step: StepRevisionPool
) -> str:
    scores_str: list[str] = []
    for s_i, s in enumerate(scores):
        inner: list[str] = []
        for e_i, e in enumerate(s.example_scores):
            inner.append(
                f"The human got the grid for Training Example {e_i + 1} {round(e.score * 100)}% correct with these instructions."
            )
        inner_text = "\n".join(inner)
        scores_str.append(
            f"""
<instructions_{s_i + 1}>
{s.instructions}
</instructions_{s_i + 1}>
<scores_from_instructions_{s_i + 1}>
{inner_text}
</scores_from_instructions_{s_i + 1}>
        """.strip()
        )

    scores_joined = "\n".join(scores_str)
    many_prompt = f"{SYNTHESIS_PROMPT}\n\n{scores_joined}"

    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": INTUITIVE_PROMPT},
                *contents_from_challenge(
                    training_examples=c.train,
                    training_example_attempts=None,
                    test_inputs=c.test,
                    include_base64=step.include_base64,
                    use_diffs=step.use_diffs,
                ),
                {
                    "type": "input_text",
                    "text": many_prompt,
                },
            ],
        },
    ]

    return (
        await get_next_structure(
            structure=ReviseInstructionsResponse,
            messages=messages,
            llm=step.instruction_llm,
        )
    ).revised_instructions


class Guess(BaseModel):
    grids: list[GRID]
    instructions_scores: list[InstructionsScore]
    llm: str

    async def save_to_db(self, avg_score: float, scores: list[float]) -> None:
        # Get database connection string from environment variable
        if "NEON_DSN" not in os.environ:
            return None

        database_url = os.environ["NEON_DSN"]

        try:
            conn = await asyncpg.connect(database_url)
        except Exception as e:
            log.warning(
                "Skipping guess DB save (could not connect)",
                error_type=type(e).__name__,
                error_message=str(e) or repr(e),
            )
            return

        try:
            await conn.execute(
                """
                INSERT INTO guess (id, grids, instructions_score_id, model, avg_score, scores)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                str(uuid.uuid4()),  # Generate new UUID for this guess
                json.dumps(self.grids),  # Convert grids to JSON string for JSONB
                self.instructions_scores[
                    0
                ].id,  # FK: schema has one id; use first test case
                self.llm,
                avg_score,
                json.dumps(scores),  # Convert scores list to JSON string for JSONB
            )
        except Exception as e:
            log.warning(
                "Skipping guess DB save (insert failed)",
                error_type=type(e).__name__,
                error_message=str(e) or repr(e),
            )
        finally:
            await conn.close()


async def get_diverse_attempts(
    c: Challenge,
    step: Step | StepRevision | StepRevisionPool,
    test_input: TestExample,
    scores: list[InstructionsScore],
    config: RunConfig,
) -> tuple[tuple[GRID, InstructionsScore], tuple[GRID, InstructionsScore]]:
    scores = sorted(scores, key=lambda x: x.score, reverse=True)
    perfect_scores = [s for s in scores if s.score == 1]
    if perfect_scores:
        scores_to_use = perfect_scores[0 : config.final_follow_times]
        if len(scores_to_use) < config.final_follow_times:
            for i in range(config.final_follow_times - len(scores_to_use)):
                scores_to_use.append(perfect_scores[i % len(perfect_scores)])
    else:
        # if no perfect scores, take the top two and use them only
        if config.final_follow_times == 1:
            scores_to_use = [scores[0]]
        else:
            top_score = scores[0]
            second_score = scores[1] if len(scores) > 1 else scores[0]

            # Calculate split - first half gets the extra spot if odd
            first_half_count = (config.final_follow_times + 1) // 2
            second_half_count = config.final_follow_times - first_half_count

            # Fill with top score first, then second score
            scores_to_use = [top_score] * first_half_count + [
                second_score
            ] * second_half_count

    futures = []
    for score_to_use in scores_to_use:
        futures.append(
            output_grid_from_instructions(
                instructions=score_to_use.instructions,
                llm=config.final_follow_llm,
                include_base64=step.include_base64,
                training_examples=c.train,
                test_input_grid=test_input.input,
                use_diffs=step.use_diffs,
                is_perfect=score_to_use.score == 1,
            )
        )
    log.debug("scores to use for final grids", scores=scores_to_use)
    final_output_grids = await asyncio.gather(*futures, return_exceptions=True)
    final_output_grids = filter_out_exceptions(
        final_output_grids, "Exception in get_diverse_attempts (final output grids)"
    )
    if not final_output_grids:
        log.error(f"No final output grids found for {c.task_id}")

    first_grid = final_output_grids[0]
    for g in final_output_grids:
        if g[0] != first_grid[0]:
            return first_grid, g
    return first_grid, first_grid


async def return_answer(
    c: Challenge,
    scores: list[InstructionsScore],
    config: RunConfig,
    step: Step | StepRevision | StepRevisionPool,
) -> tuple[Guess, Guess]:
    log.info(
        "Perfect score achieved or ending, generating final answers",
        score=scores[0].score,
    )
    futures = []
    for test in c.test:
        # take 2 attempts
        futures.append(
            get_diverse_attempts(
                c=c, step=step, test_input=test, config=config, scores=scores
            )
        )

    log.debug("Generating final output grid tuples")
    # this is a list of tuple grids, corresponding to the index of the test
    final_output_grids: list[
        tuple[tuple[GRID, InstructionsScore], tuple[GRID, InstructionsScore]]
    ] = await asyncio.gather(*futures)
    log.debug("Final grids generated", grid_count=len(final_output_grids))

    first_prediction: list[tuple[GRID, InstructionsScore]] = []
    second_prediction: list[tuple[GRID, InstructionsScore]] = []

    for i in range(len(c.test)):
        first_prediction.append(final_output_grids[i][0])
        second_prediction.append(final_output_grids[i][1])

    first_prediction_guess = Guess(
        instructions_scores=[g[1] for g in first_prediction],
        llm=config.final_follow_llm,
        grids=[g[0] for g in first_prediction],
    )
    second_prediction_guess = Guess(
        instructions_scores=[g[1] for g in second_prediction],
        llm=config.final_follow_llm,
        grids=[g[0] for g in second_prediction],
    )

    return first_prediction_guess, second_prediction_guess


async def get_answer_grids(*, c: Challenge, config: RunConfig) -> tuple[Guess, Guess]:
    if os.environ.get("VIZ", "0") == "1":
        c.viz()
    instruction_scores = []
    prev_step = config.steps[0]
    for step in config.steps:
        with log.span("step starting", step=step):
            if isinstance(step, Step):
                instruction_scores.extend(await get_instruction_scores(c=c, step=step))
            else:
                futures = []
                if isinstance(step, StepRevision):
                    for score in instruction_scores[0 : step.top_scores_used]:
                        for _ in range(step.times_per_top_score):
                            futures.append(
                                score.get_revised_instructions(c=c, step=step)
                            )
                elif isinstance(step, StepRevisionPool):
                    for _ in range(step.times):
                        if instruction_scores:
                            futures.append(
                                get_pooling_instruction_from_scores(
                                    c=c,
                                    scores=instruction_scores[0 : step.top_scores_used],
                                    step=step,
                                )
                            )
                        else:
                            log.error("Cannot do pooling with no instruction scores")
                else:
                    raise Exception(f"invalid step: {step}")

                revised_instructions = await asyncio.gather(
                    *futures, return_exceptions=True
                )
                revised_instructions = filter_out_exceptions(
                    lst=revised_instructions,
                    description="Exception in get_answer_grids (revised instructions)",
                )
                futures = []
                for revised_instruction in revised_instructions:
                    log.debug("Revised instruction", instruction=revised_instruction)
                    futures.append(
                        get_score_from_instructions(
                            c=c, instructions=revised_instruction, step=step
                        )
                    )
                new_instruction_scores = await asyncio.gather(
                    *futures, return_exceptions=True
                )
                if new_instruction_scores:
                    new_instruction_scores = filter_out_exceptions(
                        lst=new_instruction_scores,
                        description="Exception in get_answer_grids (new instruction scores)",
                    )
                    log.debug(
                        "Revised scores",
                        scores=[s.score for s in new_instruction_scores],
                    )
                    if new_instruction_scores:
                        top_revised_score = max(s.score for s in new_instruction_scores)
                        if instruction_scores:
                            log.info(
                                "Revision improvement",
                                from_score=instruction_scores[0].score,
                                to_score=top_revised_score,
                                improvement=top_revised_score
                                - instruction_scores[0].score,
                            )
                        instruction_scores = [
                            *new_instruction_scores,
                            *instruction_scores,
                        ]

            instruction_scores: list[InstructionsScore] = sorted(
                instruction_scores, key=lambda x: x.score, reverse=True
            )
            log.debug("Current scores", scores=[s.score for s in instruction_scores])
            if instruction_scores and instruction_scores[0].score == 1:
                return await return_answer(
                    c=c,
                    scores=instruction_scores,
                    config=config,
                    step=prev_step,
                )

    # TODO here we do GREG's induction but for now just do the bets score
    if instruction_scores:
        return await return_answer(
            c=c,
            scores=instruction_scores,
            config=config,
            step=prev_step,
        )
    else:
        log.error("No instruction scores found")
        raise Exception("no scores...")


SOLUTIONS_D: dict[str, list[ChallengeSolution]] = {}


def _attempts_layout_for_run_dir(
    results_run_dir: Path, challenges_path: Path
) -> tuple[Path, Path]:
    """Default aggregate path and per-task directory under a run folder."""
    attempts_subdir = results_run_dir / "attempts"
    aggregate_name = challenges_path.name.replace("_challenges", "_attempts")
    default_aggregate = attempts_subdir / aggregate_name
    if not attempts_subdir.is_dir():
        return default_aggregate, attempts_subdir
    candidates = sorted(attempts_subdir.glob("*_attempts.json"))
    if len(candidates) == 1:
        return candidates[0], attempts_subdir
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple aggregate *_attempts.json files under {attempts_subdir}; "
            "pass attempts_path and temp_attempts_dir explicitly."
        )
    return default_aggregate, attempts_subdir


def load_resume_solutions(
    attempts_path: Path,
    temp_attempts_dir: Path,
) -> dict[str, list[ChallengeSolution]]:
    """Load completed tasks from the aggregate file and/or per-task JSON files."""
    merged: dict[str, list[ChallengeSolution]] = {}
    adapter_dict = TypeAdapter(dict[str, list[ChallengeSolution]])
    adapter_list = TypeAdapter(list[ChallengeSolution])

    if attempts_path.is_file():
        try:
            merged.update(
                adapter_dict.validate_json(attempts_path.read_text(encoding="utf-8"))
            )
        except Exception as e:
            log.warning(
                "Could not parse aggregate attempts for resume",
                path=str(attempts_path),
                error=str(e),
            )

    if temp_attempts_dir.is_dir():
        aggregate_resolved = attempts_path.resolve()
        for p in sorted(temp_attempts_dir.glob("*.json")):
            if p.resolve() == aggregate_resolved:
                continue
            tid = p.stem
            if tid in merged:
                continue
            try:
                data = adapter_list.validate_json(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data:
                merged[tid] = data

    return merged


def _challenge_needs_solve(
    c: Challenge, saved: dict[str, list[ChallengeSolution]]
) -> bool:
    got = saved.get(c.task_id)
    if got is None:
        return True
    return len(got) != len(c.test)


async def solve_challenge(
    c: Challenge,
    attempts_path: Path,
    temp_attempts_dir: Path,
    solution_grids: list[GRID] | None,
    config: RunConfig,
) -> float:
    if os.getenv("USE_TASK_ID", "0") == "1":
        task_id_to_use = c.task_id
    else:
        # totally hide task id so there is no proprietary info being sent
        task_id_to_use = random_str(6)
    set_task_id(task_id_to_use)
    log.info("Starting challenge")

    with log.span("solve_challenge"):
        first_guess_obj, second_guess_obj = await get_answer_grids(c=c, config=config)
    # now write these to attempts path

    challenge_solutions: list[ChallengeSolution] = []
    for i in range(len(c.test)):
        challenge_solutions.append(
            ChallengeSolution(
                attempt_1=first_guess_obj.grids[i], attempt_2=second_guess_obj.grids[i]
            )
        )
    SOLUTIONS_D[c.task_id] = challenge_solutions

    open(temp_attempts_dir / f"{c.task_id}.json", "w").write(
        json.dumps(
            TypeAdapter(list[ChallengeSolution]).dump_python(SOLUTIONS_D[c.task_id])
        )
    )

    open(attempts_path, "w").write(
        json.dumps(
            TypeAdapter(dict[str, list[ChallengeSolution]]).dump_python(SOLUTIONS_D)
        )
    )

    if solution_grids:
        final_scores: list[float] = []
        for guess_obj in [first_guess_obj, second_guess_obj]:
            correct = 0
            total = len(solution_grids)
            guess_scores: list[float] = []
            for i in range(len(solution_grids)):
                answer_grid = guess_obj.grids[i]
                solution_grid = solution_grids[i]
                if answer_grid == solution_grid:
                    correct += 1
                    log.debug(f"Grid {i} matches")
                    guess_scores.append(1)
                else:
                    if os.getenv("LOG_GRIDS", "0") == "1":
                        log.debug(
                            f"Grid {i} mismatch",
                            expected=solution_grid,
                            actual=answer_grid,
                        )
                    guess_scores.append(0)

            score = correct / total
            await guess_obj.save_to_db(scores=guess_scores, avg_score=score)
            final_scores.append(score)
            log.info(
                "Guess result",
                score_percent=f"{round(score * 100)}%",
                correct=correct,
                total=total,
            )
        max_score = max(final_scores)
        log.info("Challenge completed", final_score=max_score)
        return max_score
    else:
        return -1


async def solve_challenges(
    challenges: list[Challenge],
    solution_grids_list: list[list[GRID]] | None,
    config: RunConfig,
    attempts_path: Path,
    temp_attempts_dir: Path,
) -> float:
    # Create semaphore to limit concurrent tasks
    semaphore = MonitoredSemaphore(config.max_concurrent_tasks, name="run_semaphore")

    async def run_one_challenge(
        *,
        challenge: Challenge,
        solution_grids: list[GRID] | None,
    ) -> float | None:
        async with semaphore:
            try:
                return await solve_challenge(
                    c=challenge,
                    solution_grids=solution_grids,
                    config=config,
                    attempts_path=attempts_path,
                    temp_attempts_dir=temp_attempts_dir,
                )
            except Exception as e:
                error_kwargs: dict[str, T.Any] = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "task_id": challenge.task_id,
                }
                if str(e) != "no scores...":
                    error_kwargs["traceback"] = "".join(
                        traceback.format_exception(type(e), e, e.__traceback__)
                    )
                log.error(
                    f"Exception in solve_challenges: {type(e).__name__}",
                    **error_kwargs,
                )
                return None

    if solution_grids_list is None:
        solution_grids_list = [[] for _ in challenges]
    coros = [
        run_one_challenge(challenge=ch, solution_grids=sg)
        for ch, sg in zip(challenges, solution_grids_list, strict=True)
    ]
    raw_scores = await tqdm_async.gather(
        *coros,
        total=len(coros),
        desc="Challenges",
        unit="task",
        dynamic_ncols=True,
    )
    scores = [s for s in raw_scores if s is not None]
    if scores:
        final_score = sum(scores) / len(scores)
        log.info(
            "Overall results",
            final_score_percent=f"{final_score * 100:.2f}%",
            total_score=sum(scores),
            challenge_count=len(scores),
        )
        return final_score
    log.error("No scores for challenges", config=config.model_dump())
    return 0


async def run_from_json(
    *,
    challenges_path: Path,
    config: RunConfig,
    truth_solutions_path: Path,
    limit: int | None,
    offset: int = 0,
    task_ids: set[str] | None = None,
    results_run_dir: Path | None = None,
    attempts_path: Path | None = None,
    temp_attempts_dir: Path | None = None,
    resume_dir: Path | None = None,
) -> Path:
    global SOLUTIONS_D

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    resuming = resume_dir is not None
    if resuming:
        results_run_dir = resume_dir.resolve()
        if not results_run_dir.is_dir():
            raise NotADirectoryError(
                f"Resume directory does not exist: {results_run_dir}"
            )
    elif results_run_dir is None:
        results_run_dir = results_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_run_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_run_dir = results_run_dir.resolve()
        results_run_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("LOG_FILE"):
        configure_local_log_path(results_run_dir / "arc.log")

    if attempts_path is None and temp_attempts_dir is None:
        attempts_path, temp_attempts_dir = _attempts_layout_for_run_dir(
            results_run_dir, challenges_path
        )
        temp_attempts_dir.mkdir(parents=True, exist_ok=True)
    elif attempts_path is None or temp_attempts_dir is None:
        raise ValueError(
            "attempts_path and temp_attempts_dir must both be set or both omitted"
        )

    if resuming:
        SOLUTIONS_D = load_resume_solutions(attempts_path, temp_attempts_dir)
    else:
        SOLUTIONS_D = {}

    run_id = generate_run_id()
    print(f"\n{'=' * 50}")
    if resuming:
        print(f"Resuming run (session id: {run_id})")
        print(f"Results directory: {results_run_dir}")
        print(f"Loaded {len(SOLUTIONS_D)} task(s) from previous output")
    else:
        print(f"Starting new run with ID: {run_id}")
        print(f"Results directory: {results_run_dir}")
    print(f"{'=' * 50}\n")

    raw_challenges: dict[str, dict] = json.loads(challenges_path.read_text())
    root_challenges: dict[str, Challenge] = {
        k: Challenge.model_validate({**v, "task_id": k})
        for k, v in raw_challenges.items()
    }
    if task_ids:
        root_challenges = {k: v for k, v in root_challenges.items() if k in task_ids}
    challenges_list = list(root_challenges.values())
    root_solutions = TypeAdapter(dict[str, list[list[list[int]]]]).validate_json(
        truth_solutions_path.read_text()
    )

    if resuming:
        ordered = sorted(challenges_list, key=lambda x: len(str(x)))
        incomplete = [c for c in ordered if _challenge_needs_solve(c, SOLUTIONS_D)]
        if limit is not None:
            challenges_list = incomplete[offset : offset + limit]
            print(
                f"Resume: {len(incomplete)} incomplete challenges "
                f"offset={offset}, limit={limit} → {len(challenges_list)} to run."
            )
        else:
            challenges_list = incomplete
            print(f"Resume: {len(challenges_list)} incomplete challenges (no limit).")
    else:
        if limit:
            challenges_list = challenges_list[offset : offset + limit]
        challenges_list = sorted(challenges_list, key=lambda x: len(str(x)))

    solutions_list = [root_solutions[c.task_id] for c in challenges_list]

    log.info(
        "Starting run",
        config=config.model_dump(),
        challenges_path=str(challenges_path),
        num_challenges=len(root_challenges),
        results_run_dir=str(results_run_dir),
        resuming=resuming,
        challenges_to_run=len(challenges_list),
    )

    temp_attempts_dir.mkdir(exist_ok=True, parents=True)

    if not challenges_list:
        log.info(
            "No challenges to run",
            resuming=resuming,
            results_run_dir=str(results_run_dir),
        )
        print("Nothing to run; no challenges queued.")
    else:
        final_scores = await solve_challenges(
            challenges=challenges_list,
            attempts_path=attempts_path,
            solution_grids_list=solutions_list,
            config=config,
            temp_attempts_dir=temp_attempts_dir,
        )
        log.info("Run completed", final_scores=final_scores)

    return attempts_path


async def run() -> None:
    # Generate and print run ID at the start

    year = "2024"
    train_or_eval = "evaluation"
    root_dir = Path(__file__).parent.parent

    challenges_path = (
        root_dir
        / "data"
        / f"arc-prize-{year}"
        / f"arc-agi_{train_or_eval}_challenges.json"
    )

    solutions_path = (
        root_dir
        / "data"
        / f"arc-prize-{year}"
        / f"arc-agi_{train_or_eval}_solutions.json"
    )
    # solutions_path = None

    parser = argparse.ArgumentParser(description="Run ARC solver")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="LLM handle provider/model_id, e.g. openai/gpt-5.2 or openrouter/qwen/qwen3.5",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="DIR",
        help="Existing results/<timestamp> folder to resume (reuses attempts/ and arc.log; skips tasks already saved)",
    )
    args = parser.parse_args()

    from src.configs.gpt52_configs import gpt52_config_prod

    run_config = gpt52_config_prod
    if args.model:
        try:
            parse_llm(args.model)
        except ValueError as e:
            raise ValueError(str(e)) from e
        run_config = with_llm(run_config, args.model)

    attempts_aggregate = await run_from_json(
        challenges_path=challenges_path,
        truth_solutions_path=solutions_path,
        config=run_config,
        limit=None,
        offset=0,
        # task_ids={},
        resume_dir=args.resume,
    )

    if solutions_path:
        evaluate_solutions(
            attempts_solutions_path=attempts_aggregate,
            truth_solutions_path=solutions_path,
        )


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
