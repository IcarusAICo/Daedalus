"""extract_text: OCR service backed by GLM-OCR (zai-org/GLM-OCR).

Loads the model on ``start``, processes image batches via ``query``,
and unloads on ``stop``.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from daedalus.core import ExecutionContext, ServiceSkill, SkillSpec, register
from daedalus.core.spec import SkillExample, SkillVersion

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "zai-org/GLM-OCR"


class ExtractTextStartInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str = Field(
        default=_DEFAULT_MODEL,
        description="HuggingFace model id or local path to GLM-OCR weights.",
    )
    device: str = Field(
        default="auto",
        description="Torch device map ('auto', 'cpu', 'cuda', 'cuda:0', etc.).",
    )
    max_new_tokens: int = Field(default=8192, ge=64, le=32768)


class ImageItem(BaseModel):
    """One image to OCR — either a file path or base64-encoded PNG/JPEG."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(default=None, description="Filesystem path to image.")
    base64: str | None = Field(default=None, description="Base64-encoded image bytes.")
    prompt: str = Field(
        default="Text Recognition:",
        description="Task prompt (e.g. 'Text Recognition:', 'Table Recognition:').",
    )


class ExtractTextQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images: list[ImageItem] = Field(min_length=1, max_length=128)


class ExtractedText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    index: int


class ExtractTextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[ExtractedText]


@register
class ExtractText(ServiceSkill):
    SPEC = SkillSpec(
        id="extract_text",
        version=SkillVersion(raw="0.1.0"),
        kind="service",
        description=(
            "OCR service using GLM-OCR. Load the model once with start(), "
            "extract text from image batches via query(), release with stop()."
        ),
        side_effects=["llm_call"],
        preconditions=[],
        examples=[
            SkillExample(
                inputs={"model_path": "zai-org/GLM-OCR", "device": "auto"},
            ),
        ],
        tags=["service", "ocr", "vision", "ml"],
    )
    Inputs = ExtractTextStartInput
    Outputs = ExtractTextOutput
    QueryInputs = ExtractTextQueryInput

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "auto"
        self._max_new_tokens: int = 8192

    def start(self, inputs: ExtractTextStartInput, ctx: ExecutionContext) -> None:  # type: ignore[override]
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "extract_text requires the 'transformers' package. "
                "Install with: pip install 'transformers @ git+https://github.com/huggingface/transformers.git'"
            ) from exc

        log.info("Loading GLM-OCR model from %s ...", inputs.model_path)
        self._processor = AutoProcessor.from_pretrained(inputs.model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=inputs.model_path,
            torch_dtype="auto",
            device_map=inputs.device,
        )
        self._device = inputs.device
        self._max_new_tokens = inputs.max_new_tokens
        log.info("GLM-OCR model loaded successfully.")

    def query(self, inputs: ExtractTextQueryInput, ctx: ExecutionContext) -> ExtractTextOutput:  # type: ignore[override]
        if self._model is None or self._processor is None:
            raise RuntimeError("Service not started; call start() first.")

        results: list[ExtractedText] = []
        for idx, item in enumerate(inputs.images):
            text = self._process_single(item)
            results.append(ExtractedText(text=text, index=idx))

        return ExtractTextOutput(results=results)

    def stop(self, ctx: ExecutionContext) -> None:
        import gc

        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        log.info("GLM-OCR model unloaded.")

    def _process_single(self, item: ImageItem) -> str:
        image_source: dict[str, Any]
        if item.path:
            image_source = {"type": "image", "url": item.path}
        elif item.base64:
            data_uri = f"data:image/png;base64,{item.base64}"
            image_source = {"type": "image", "url": data_uri}
        else:
            raise ValueError("ImageItem must have either 'path' or 'base64' set.")

        messages = [
            {
                "role": "user",
                "content": [
                    image_source,
                    {"type": "text", "text": item.prompt},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        inputs.pop("token_type_ids", None)

        generated_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        output_text = self._processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return output_text.strip()
