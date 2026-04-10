from pydantic import BaseModel

GRID = list[list[int]]


class Example(BaseModel):
    input: GRID
    output: GRID


class TestExample(BaseModel):
    input: GRID
    output: GRID | None = None


class Challenge(BaseModel):
    task_id: str

    train: list[Example]
    test: list[TestExample]

    def __hash__(self) -> int:
        return hash(self.model_dump_json(include={"train", "test"}))

    def size(self) -> int:
        return len(self.model_dump_json())

    def viz(
        self,
        *,
        train_attempts: list[GRID] | None = None,
        solutions: list[GRID] | None = None,
        test_attempts: list[GRID] | None = None,
    ) -> None:
        from src.viz import viz_many

        if train_attempts:
            assert len(train_attempts) == len(self.train)
        if test_attempts:
            assert len(test_attempts) == len(self.test)

        grids: list[list[GRID]] = []
        row_colors: list[str] = []

        for i, train in enumerate(self.train):
            row = [train.input, train.output]
            row_color = "white"
            if train_attempts:
                attempt = train_attempts[i]
                if train.output == attempt:
                    row_color = "green"
                else:
                    row_color = "red"
                row.append(attempt)
            grids.append(row)
            row_colors.append(row_color)

        for i, test in enumerate(self.test):
            row = [test.input]
            row_color = "white"
            solution = None
            if solutions:
                solution = solutions[i]
                row.append(solution)
            if test_attempts:
                attempt = test_attempts[i]
                if solution:
                    if solution == attempt:
                        row_color = "green"
                    else:
                        row_color = "red"
                row.append(attempt)
            grids.append(row)
            row_colors.append(row_color)
        try:
            viz_many(grids=grids, color_map=COLOR_MAP, row_border_colors=row_colors)
        except Exception as e:
            print("EEEE in viz", e)

    def to_basic_prompt(self, use_cot: bool) -> list[str]:
        prompt = "Find the common rule that maps an input grid to an output grid, given the examples below."
        cot_line = "Think step by step and give your reasoning in reasoning tags: <reasoning>xxx</reasoning>."
        cot = "" if not use_cot else f" {cot_line}"
        test_str = f"Below is a test input grid. Predict the corresponding output grid by applying the rule you found.{cot} Your final answer should just be the text output grid itself."
        example_strs: list[str] = []
        for i, example in enumerate(self.train):
            _input_str = self.grid_to_str(grid=example.input)
            output_str = self.grid_to_str(grid=example.output)
            example_strs.append(
                f"Example {i + 1}:\n\nInput:\n{_input_str}\nOutput:\n{output_str}"
            )
        example_str = "\n\n".join(example_strs)
        final_strs: list[str] = []
        for test_example in self.test:
            final_strs.append(
                f"""
{prompt}

{example_str}

{test_str}

Input:
{self.grid_to_str(grid=test_example.input)}
                """.strip()
            )
        return final_strs

    @staticmethod
    def grid_from_str(s: str) -> GRID:
        def is_int(token: str) -> bool:
            try:
                int(token)
                return True
            except ValueError:
                return False

        lines = s.splitlines()
        final_grid = []
        current_grid = []

        for line in lines:
            tokens = line.split()
            # Check if line is non-empty and every token is an integer.
            if tokens and all(is_int(token) for token in tokens):
                current_grid.append([int(token) for token in tokens])
            else:
                # If we hit a non-grid line, and we already accumulated some grid rows,
                # assume the current grid block has ended.
                if current_grid:
                    final_grid = current_grid
                    current_grid = []

        # If the string ended with grid rows, update final_grid.
        if current_grid:
            final_grid = current_grid

        return final_grid

    @staticmethod
    def grid_to_str(grid: GRID) -> str:
        return "\n".join(" ".join(str(x) for x in row) for row in grid)

    @staticmethod
    def grid_to_base64(grid: GRID) -> str:
        from src.viz import base64_from_grid

        return base64_from_grid(grid=grid)


COLOR_MAP: dict[int, str] = {
    0: "black",
    1: "#0074D9",  # blue
    2: "#FF4136",  # red
    3: "#2ECC40",  # green
    4: "#FFDC00",  # yellow
    5: "#AAAAAA",  # grey
    6: "#F012BE",  # pink
    7: "#FF851B",  # orange
    8: "#9d00ff",  # purple
    9: "#870C25",  # brown
}
