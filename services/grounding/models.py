"""Pydantic models for the grounding service API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    """Full-screen parse request: detect all UI elements."""

    image_b64: str = Field(description="Base64-encoded PNG screenshot.")


class UIElement(BaseModel):
    """A single detected UI element."""

    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 0.0
    ocr_text: str = ""
    caption: str = ""

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class ParseResponse(BaseModel):
    """All detected UI elements from a screenshot."""

    elements: list[UIElement]
    parse_time_ms: float


class LocateRequest(BaseModel):
    """Locate a specific element by description."""

    image_b64: str = Field(description="Base64-encoded PNG screenshot.")
    description: str = Field(description="Natural-language description, e.g. 'submit button'.")
    mode: Literal["point", "box", "all"] = "point"
    confidence_threshold: float = 0.3
    target_width: int | None = Field(
        default=None,
        description="If set, return coordinates mapped to this width instead of image width.",
    )
    target_height: int | None = Field(
        default=None,
        description="If set, return coordinates mapped to this height instead of image height.",
    )


class LocateMatch(BaseModel):
    """A single match from the locate endpoint."""

    label: str
    x: int
    y: int
    box: tuple[int, int, int, int] | None = None
    confidence: float


class LocateResponse(BaseModel):
    """Response from the locate endpoint."""

    found: bool
    matches: list[LocateMatch]
    locate_time_ms: float
