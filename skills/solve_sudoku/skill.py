"""solve_sudoku: solve a 9x9 Sudoku puzzle using backtracking."""

from __future__ import annotations

import copy
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion


class SudokuInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grid: List[List[int]] = Field(
        description="9x9 grid of integers. 0 = empty cell, 1-9 = filled cell."
    )

    @field_validator("grid")
    @classmethod
    def validate_grid(cls, v: List[List[int]]) -> List[List[int]]:
        if len(v) != 9:
            raise ValueError(f"Grid must have exactly 9 rows, got {len(v)}")
        for i, row in enumerate(v):
            if len(row) != 9:
                raise ValueError(f"Row {i} must have exactly 9 cells, got {len(row)}")
            for j, cell in enumerate(row):
                if not (0 <= cell <= 9):
                    raise ValueError(
                        f"Cell ({i},{j}) value {cell} is out of range [0,9]"
                    )
        return v


class SudokuOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solved: Optional[List[List[int]]] = Field(
        description="Solved 9x9 grid, or null if no solution exists."
    )
    is_valid_input: bool = Field(
        description="True if the input grid is a valid (possibly partial) Sudoku."
    )


def _is_valid_board(grid: List[List[int]]) -> bool:
    """Check that the filled cells don't violate Sudoku constraints."""
    for i in range(9):
        row_vals = [grid[i][j] for j in range(9) if grid[i][j] != 0]
        if len(row_vals) != len(set(row_vals)):
            return False
        col_vals = [grid[r][i] for r in range(9) if grid[r][i] != 0]
        if len(col_vals) != len(set(col_vals)):
            return False
    for br in range(3):
        for bc in range(3):
            box_vals = [
                grid[br * 3 + r][bc * 3 + c]
                for r in range(3)
                for c in range(3)
                if grid[br * 3 + r][bc * 3 + c] != 0
            ]
            if len(box_vals) != len(set(box_vals)):
                return False
    return True


def _is_safe(grid: List[List[int]], row: int, col: int, num: int) -> bool:
    """Check if placing `num` at (row, col) is safe."""
    # Check row
    if num in grid[row]:
        return False
    # Check column
    if num in (grid[r][col] for r in range(9)):
        return False
    # Check 3x3 box
    br, bc = (row // 3) * 3, (col // 3) * 3
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if grid[r][c] == num:
                return False
    return True


def _find_empty(grid: List[List[int]]) -> Optional[tuple]:
    """Find the next empty cell (value == 0). Returns (row, col) or None."""
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return (r, c)
    return None


def _solve(grid: List[List[int]]) -> bool:
    """Solve in-place using backtracking. Returns True if solved."""
    pos = _find_empty(grid)
    if pos is None:
        return True  # All cells filled
    row, col = pos
    for num in range(1, 10):
        if _is_safe(grid, row, col, num):
            grid[row][col] = num
            if _solve(grid):
                return True
            grid[row][col] = 0
    return False


@register
class SolveSudoku(AtomicSkill):
    SPEC = SkillSpec(
        id="solve_sudoku",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description=(
            "Solves a 9x9 Sudoku puzzle using backtracking. "
            "Input is a 9x9 grid as a list of 9 lists of 9 integers (0 = empty). "
            "Returns the solved grid or null if unsolvable."
        ),
        side_effects=[],
        preconditions=[
            "len(grid) == 9",
            "all(len(row) == 9 for row in grid)",
            "all(0 <= cell <= 9 for row in grid for cell in row)",
        ],
        postconditions=[
            "solved is None or (len(solved) == 9 and all(len(r) == 9 for r in solved))",
        ],
        examples=[
            SkillExample(
                inputs={
                    "grid": [
                        [0, 0, 0, 0, 0, 0, 0, 0, 3],
                        [0, 0, 8, 6, 0, 1, 0, 2, 0],
                        [0, 0, 0, 3, 0, 0, 4, 0, 0],
                        [0, 7, 0, 0, 0, 0, 0, 5, 0],
                        [9, 0, 1, 2, 0, 0, 0, 0, 0],
                        [4, 0, 0, 8, 0, 6, 0, 3, 0],
                        [0, 5, 4, 7, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 6, 0],
                        [0, 0, 6, 0, 0, 4, 7, 0, 0],
                    ]
                },
                note="Classic Sudoku puzzle.",
            )
        ],
        tests=["basic.json", "unsolvable.json"],
        tags=["puzzle", "sudoku", "solver", "algorithm"],
    )
    Inputs = SudokuInput
    Outputs = SudokuOutput

    def run(self, inputs: SudokuInput, ctx: ExecutionContext) -> SudokuOutput:  # type: ignore[override]
        grid = copy.deepcopy(inputs.grid)

        # Validate the initial board
        if not _is_valid_board(grid):
            return SudokuOutput(solved=None, is_valid_input=False)

        # Attempt to solve
        if _solve(grid):
            return SudokuOutput(solved=grid, is_valid_input=True)
        else:
            return SudokuOutput(solved=None, is_valid_input=True)
