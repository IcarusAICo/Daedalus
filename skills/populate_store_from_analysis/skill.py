"""populate_store_from_analysis: write coordinate list into a RunStore table."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import AtomicSkill, ExecutionContext, SkillSpec, register
from daedalus.core.spec import SkillVersion


class CoordPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int = Field(description="X pixel coordinate.")
    y: int = Field(description="Y pixel coordinate.")


class PopulateStoreInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clicks: List[CoordPoint] = Field(
        description="Array of {x, y} coordinates from analyze_and_solve_enclosure_puzzle."
    )
    store_table: str = Field(
        default="click_targets",
        description="Name of the RunStore table to populate.",
    )


class PopulateStoreOutputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(description="Number of rows written to the store.")
    store_table: str = Field(description="Name of the RunStore table that was populated.")


@register
class PopulateStoreFromAnalysis(AtomicSkill):
    SPEC = SkillSpec(
        id="populate_store_from_analysis",
        version=SkillVersion(raw="0.1.0"),
        kind="atomic",
        description=(
            "Takes a list of {x,y} click coordinates and writes them into a "
            "RunStore table so click_all can consume them."
        ),
        side_effects=[],
        preconditions=["inputs.clicks is a list of {x, y} dicts"],
        postconditions=["store table contains all input coordinates"],
        examples=[],
        tests=["basic.json"],
        tags=["store", "data", "bridge"],
    )
    Inputs = PopulateStoreInputs
    Outputs = PopulateStoreOutputs

    def run(self, inputs: PopulateStoreInputs, ctx: ExecutionContext) -> PopulateStoreOutputs:  # type: ignore[override]
        table = inputs.store_table

        # Create the table if it doesn't already exist
        if ctx.store is not None and table not in ctx.store.table_names():
            ctx.store.create_table(table, {"x": "int", "y": "int"})

        count = 0
        for coord in inputs.clicks:
            if ctx.store is not None:
                ctx.store.append(table, {"x": coord.x, "y": coord.y})
            count += 1

        return PopulateStoreOutputs(count=count, store_table=table)
