"""Model loading and inference for the grounding service.

Uses a two-tier approach:
  1. KV-Ground-4B (primary): A Qwen3-VL-based GUI grounding VLM fine-tuned
     for high-resolution screenshots. Faster and more accurate than ZonUI-3B.
  2. OmniParser V2 (fallback/parse): YOLO icon detection + Florence-2
     captioning for full-screen element enumeration via /parse.

Legacy ZonUI-3B code is preserved but disabled (set GROUNDING_USE_ZONUI=1 to
use it instead of KV-Ground-4B).
"""

from __future__ import annotations

import ast
import base64
import io
import logging
import os
import time
from difflib import SequenceMatcher
from typing import Any

import numpy as np
from PIL import Image

from models import LocateMatch, UIElement

log = logging.getLogger(__name__)


def _decode_image(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


class GroundingEngine:
    """Manages model lifecycle and inference."""

    KVGROUND_MODEL_ID = "vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315"

    def __init__(self) -> None:
        self._yolo: Any = None
        self._caption_model: Any = None
        self._caption_processor: Any = None
        self._zonui_model: Any = None
        self._zonui_processor: Any = None
        self._zonui_tokenizer: Any = None
        self._kvground_model: Any = None
        self._kvground_processor: Any = None
        self._device: str = "cpu"
        self._loaded = False
        self._zonui_loaded = False
        self._kvground_loaded = False

    def load(self, device: str = "cuda") -> None:
        import torch

        if not torch.cuda.is_available():
            log.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        self._device = device

        use_zonui = os.environ.get("GROUNDING_USE_ZONUI", "0") == "1"

        if use_zonui:
            self._load_zonui(device)
        else:
            self._load_kvground(device)

        # Load OmniParser V2 (YOLO + Florence-2 for /parse and fallback)
        try:
            from ultralytics import YOLO
            from transformers import AutoModelForCausalLM, AutoProcessor

            log.info("loading YOLO icon detector...")
            self._yolo = YOLO("weights/icon_detect/model.pt")

            log.info("loading Florence-2 caption model on %s...", device)
            self._caption_processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-base", trust_remote_code=True,
            )
            self._caption_model = AutoModelForCausalLM.from_pretrained(
                "weights/icon_caption_florence",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(device)

            self._loaded = True
            log.info("OmniParser models loaded successfully")
        except Exception as exc:
            log.warning("failed to load OmniParser models: %s", exc)
            self._loaded = False

    def _load_kvground(self, device: str) -> None:
        """Load KV-Ground-4B-BaseGuiOwl1.5-0228 (Qwen3-VL architecture)."""
        import torch

        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            log.info("loading KV-Ground-4B grounding model on %s...", device)
            self._kvground_model = AutoModelForImageTextToText.from_pretrained(
                self.KVGROUND_MODEL_ID,
                dtype=torch.bfloat16,
                device_map="auto" if device == "cuda" else None,
            ).eval()
            self._kvground_processor = AutoProcessor.from_pretrained(
                self.KVGROUND_MODEL_ID,
            )
            self._kvground_loaded = True
            log.info("KV-Ground-4B loaded successfully")
        except Exception as exc:
            log.warning("failed to load KV-Ground-4B: %s — falling back to ZonUI-3B", exc)
            self._kvground_loaded = False
            self._load_zonui(device)

    def _load_zonui(self, device: str) -> None:
        """Load ZonUI-3B (legacy, Qwen2.5-VL architecture)."""
        import torch

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
            from transformers.generation import GenerationConfig

            log.info("loading ZonUI-3B grounding model on %s...", device)
            self._zonui_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "zonghanHZH/ZonUI-3B",
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.bfloat16,
            ).eval()
            self._zonui_processor = AutoProcessor.from_pretrained("zonghanHZH/ZonUI-3B")
            self._zonui_tokenizer = AutoTokenizer.from_pretrained(
                "zonghanHZH/ZonUI-3B", trust_remote_code=True,
            )
            gen_config = GenerationConfig.from_pretrained(
                "zonghanHZH/ZonUI-3B", trust_remote_code=True,
            )
            gen_config.max_length = 4096
            gen_config.do_sample = False
            gen_config.temperature = 0.0
            self._zonui_model.generation_config = gen_config
            self._zonui_loaded = True
            log.info("ZonUI-3B loaded successfully")
        except Exception as exc:
            log.warning("failed to load ZonUI-3B: %s — will use OmniParser only", exc)
            self._zonui_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded or self._zonui_loaded or self._kvground_loaded

    def parse(self, image_b64: str) -> tuple[list[UIElement], float]:
        t0 = time.perf_counter()

        if not self._loaded:
            return [], (time.perf_counter() - t0) * 1000

        import torch
        from torchvision.transforms import ToPILImage
        import cv2

        img = _decode_image(image_b64)
        w, h = img.size

        try:
            out = self._yolo.predict(
                img, imgsz=max(w, h), conf=0.05, iou=0.7, verbose=False,
            )[0]
        except Exception as exc:
            log.error("YOLO predict failed: %s", exc)
            return [], (time.perf_counter() - t0) * 1000

        if out.boxes is None:
            return [], (time.perf_counter() - t0) * 1000

        xyxy_bboxes = out.boxes.xyxy
        # Normalize to [0,1]
        xyxy_norm = xyxy_bboxes / torch.Tensor([w, h, w, h]).to(xyxy_bboxes.device)
        img_np = np.asarray(img)

        # Crop, caption, and OCR each detected icon
        boxes_list = xyxy_norm.tolist()
        captions = self._caption_boxes(img_np, boxes_list, w, h)
        ocr_texts = self._ocr_boxes(img_np, boxes_list, w, h)

        elements: list[UIElement] = []
        for i, bbox_norm in enumerate(boxes_list):
            x1 = int(bbox_norm[0] * w)
            y1 = int(bbox_norm[1] * h)
            x2 = int(bbox_norm[2] * w)
            y2 = int(bbox_norm[3] * h)
            conf = float(out.boxes.conf[i]) if out.boxes.conf is not None else 0.5
            caption = captions[i] if i < len(captions) else "unknown"
            ocr_text = ocr_texts[i] if i < len(ocr_texts) else ""
            # Use OCR text as the primary label when it's non-empty and
            # looks like actual text (not a generic description). The caption
            # is kept as a fallback / supplementary label.
            label = self._merge_label(caption, ocr_text)
            elements.append(UIElement(
                label=label, x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf, ocr_text=ocr_text, caption=caption,
            ))

        elapsed = (time.perf_counter() - t0) * 1000
        return elements, elapsed

    def _resize_preserve_aspect(
        self, crop: np.ndarray, target_size: int = 224,
    ) -> np.ndarray:
        """Resize a crop to target_size x target_size preserving aspect ratio.

        Scales the image so the longest side fits target_size, then pads with
        white (255) to fill the square. This avoids the text distortion caused
        by squashing non-square UI elements into a square.
        """
        import cv2

        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

        scale = target_size / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
        y_off = (target_size - new_h) // 2
        x_off = (target_size - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized[:, :, :3]
        return canvas

    def _caption_boxes(
        self, img_np: np.ndarray, boxes_norm: list[list[float]], w: int, h: int,
        batch_size: int = 64,
    ) -> list[str]:
        """Caption cropped icon regions using Florence-2."""
        import torch
        from torchvision.transforms import ToPILImage

        crops = []
        for bbox in boxes_norm:
            try:
                x1 = int(bbox[0] * w)
                y1 = int(bbox[1] * h)
                x2 = int(bbox[2] * w)
                y2 = int(bbox[3] * h)
                crop = img_np[y1:y2, x1:x2, :]
                crop = self._resize_preserve_aspect(crop, target_size=224)
                crops.append(ToPILImage()(crop))
            except Exception:
                crops.append(ToPILImage()(np.ones((224, 224, 3), dtype=np.uint8) * 255))

        if not crops:
            return []

        captions: list[str] = []
        prompt = "<CAPTION>"
        for idx in range(0, len(crops), batch_size):
            batch = crops[idx : idx + batch_size]
            inputs = self._caption_processor(
                images=batch, text=[prompt] * len(batch),
                return_tensors="pt", do_resize=False,
            )
            if self._device in {"cuda", "mps"}:
                inputs = inputs.to(device=self._device, dtype=torch.float16)

            with torch.inference_mode():
                generated_ids = self._caption_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=20, num_beams=1, do_sample=False,
                )
            texts = self._caption_processor.batch_decode(
                generated_ids, skip_special_tokens=True,
            )
            captions.extend([t.strip() for t in texts])

        return captions

    def _ocr_boxes(
        self, img_np: np.ndarray, boxes_norm: list[list[float]], w: int, h: int,
        batch_size: int = 64,
    ) -> list[str]:
        """Run OCR on cropped regions using Florence-2's <OCR> task.

        Returns the raw text detected in each box. This complements the caption
        pass by providing exact text content (labels, button text, etc.) that
        the caption model may misread or describe abstractly.
        """
        import torch
        from torchvision.transforms import ToPILImage

        crops = []
        for bbox in boxes_norm:
            try:
                x1 = int(bbox[0] * w)
                y1 = int(bbox[1] * h)
                x2 = int(bbox[2] * w)
                y2 = int(bbox[3] * h)
                crop = img_np[y1:y2, x1:x2, :]
                crop = self._resize_preserve_aspect(crop, target_size=224)
                crops.append(ToPILImage()(crop))
            except Exception:
                crops.append(ToPILImage()(np.ones((224, 224, 3), dtype=np.uint8) * 255))

        if not crops:
            return []

        ocr_texts: list[str] = []
        prompt = "<OCR>"
        for idx in range(0, len(crops), batch_size):
            batch = crops[idx : idx + batch_size]
            inputs = self._caption_processor(
                images=batch, text=[prompt] * len(batch),
                return_tensors="pt", do_resize=False,
            )
            if self._device in {"cuda", "mps"}:
                inputs = inputs.to(device=self._device, dtype=torch.float16)

            with torch.inference_mode():
                generated_ids = self._caption_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=50, num_beams=1, do_sample=False,
                )
            texts = self._caption_processor.batch_decode(
                generated_ids, skip_special_tokens=True,
            )
            ocr_texts.extend([t.strip() for t in texts])

        return ocr_texts

    def _locate_with_kvground(
        self, image_b64: str, description: str, mode: str = "point",
        target_width: int | None = None, target_height: int | None = None,
    ) -> tuple[list[LocateMatch], float] | None:
        """Use KV-Ground-8B to locate an element by description.

        The model outputs coordinates in the processor's internal resolution.
        We map them to the target coordinate space:
          - If target_width/target_height are set, output is in that space
          - Otherwise output is in the original image's pixel space
        Returns None if KV-Ground is not loaded or fails.
        """
        if not self._kvground_loaded:
            return None

        import torch

        t0 = time.perf_counter()
        img = _decode_image(image_b64)
        orig_w, orig_h = img.size

        # The output coordinate space — either the caller's target or the image itself
        out_w = target_width if target_width else orig_w
        out_h = target_height if target_height else orig_h

        if mode == "box":
            system_prompt = (
                "Based on the screenshot of the page, I give a text description and you "
                "give the bounding box of the described element. Return the top-left and "
                "bottom-right coordinates as [[x1, y1], [x2, y2]]."
            )
        else:
            system_prompt = (
                "Based on the screenshot of the page, I give a text description and you "
                "give its corresponding location. The coordinate represents a clickable "
                "location [x, y] for an element."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image", "image": img},
                    {"type": "text", "text": description},
                ],
            }
        ]

        try:
            text = self._kvground_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._kvground_processor(
                text=[text], images=[img],
                return_tensors="pt",
            ).to(self._kvground_model.device)

            # KV-Ground outputs coordinates normalized to [0, 1000] range.
            # We don't need model_w/model_h — just divide by 1000 and multiply
            # by the output coordinate space.
            ip = self._kvground_processor.image_processor
            merge_size = ip.merge_size
            patch_size = ip.patch_size
            image_grid_thw = inputs.get("image_grid_thw")
            if image_grid_thw is not None and len(image_grid_thw) > 0:
                _, grid_h, grid_w = image_grid_thw[0].tolist()
                model_w = int(grid_w * merge_size * patch_size)
                model_h = int(grid_h * merge_size * patch_size)
            else:
                model_w = orig_w
                model_h = orig_h

            with torch.inference_mode():
                generated_ids = self._kvground_model.generate(
                    **inputs, max_new_tokens=50, do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._kvground_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            log.info("KV-Ground raw output for %r (mode=%s): %s (model res: %dx%d, output space: %dx%d)",
                     description, mode, output_text, model_w, model_h, out_w, out_h)

            coordinates = ast.literal_eval(output_text)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Model outputs in [0, 1000] normalized space
            return self._parse_grounding_coordinates(
                coordinates, output_text, out_w, out_h, description, mode, elapsed_ms,
                model_w=1000, model_h=1000,
            )

        except Exception as exc:
            log.warning("KV-Ground locate failed for %r: %s", description, exc)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return None

    def _parse_grounding_coordinates(
        self,
        coordinates: Any,
        raw_output: str,
        orig_w: int,
        orig_h: int,
        description: str,
        mode: str,
        elapsed_ms: float,
        model_w: int | None = None,
        model_h: int | None = None,
    ) -> tuple[list[LocateMatch], float] | None:
        """Parse coordinate output from a grounding model into LocateMatch results.

        Coordinates from the model are in the model's internal resolution
        (model_w x model_h). We normalize to [0,1] then scale to original.

        Handles multiple output formats:
          - [x, y] point
          - [x1, y1, x2, y2] flat bounding box
          - [[x1, y1], [x2, y2]] nested bounding box
        """
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            log.warning("Unexpected coordinate format: %s", raw_output)
            return None

        # Scale factors: model coords -> original image coords
        sx = orig_w / model_w if model_w else 1.0
        sy = orig_h / model_h if model_h else 1.0

        if mode == "box":
            return self._parse_box_coordinates(
                coordinates, raw_output, orig_w, orig_h, description, elapsed_ms, sx, sy,
            )

        # Point mode — but model may return a bbox anyway
        if len(coordinates) == 2 and not isinstance(coordinates[0], list):
            abs_x = max(0, min(int(coordinates[0] * sx), orig_w - 1))
            abs_y = max(0, min(int(coordinates[1] * sy), orig_h - 1))
            match = LocateMatch(
                label=description, x=abs_x, y=abs_y,
                box=None, confidence=0.9,
            )
            return [match], elapsed_ms

        if len(coordinates) == 4 and not isinstance(coordinates[0], list):
            # Model returned [x1, y1, x2, y2] in point mode — take center
            x1 = max(0, min(int(coordinates[0] * sx), orig_w - 1))
            y1 = max(0, min(int(coordinates[1] * sy), orig_h - 1))
            x2 = max(0, min(int(coordinates[2] * sx), orig_w - 1))
            y2 = max(0, min(int(coordinates[3] * sy), orig_h - 1))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            match = LocateMatch(
                label=description, x=cx, y=cy,
                box=(x1, y1, x2, y2), confidence=0.9,
            )
            return [match], elapsed_ms

        if (len(coordinates) == 2 and isinstance(coordinates[0], list)):
            # [[x1,y1],[x2,y2]] in point mode — take center
            x1 = max(0, min(int(coordinates[0][0] * sx), orig_w - 1))
            y1 = max(0, min(int(coordinates[0][1] * sy), orig_h - 1))
            x2 = max(0, min(int(coordinates[1][0] * sx), orig_w - 1))
            y2 = max(0, min(int(coordinates[1][1] * sy), orig_h - 1))
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            match = LocateMatch(
                label=description, x=cx, y=cy,
                box=(x1, y1, x2, y2), confidence=0.9,
            )
            return [match], elapsed_ms

        log.warning("Unexpected point format: %s", raw_output)
        return None

    def _parse_box_coordinates(
        self,
        coordinates: list,
        raw_output: str,
        orig_w: int,
        orig_h: int,
        description: str,
        elapsed_ms: float,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> tuple[list[LocateMatch], float] | None:
        """Parse box-mode coordinate output."""
        if len(coordinates) == 2 and isinstance(coordinates[0], list):
            # [[x1,y1],[x2,y2]]
            rx1, ry1 = coordinates[0]
            rx2, ry2 = coordinates[1]
        elif len(coordinates) == 4 and not isinstance(coordinates[0], list):
            rx1, ry1, rx2, ry2 = coordinates
        elif len(coordinates) == 2 and not isinstance(coordinates[0], list):
            # Only a point returned in box mode
            abs_x = max(0, min(int(coordinates[0] * sx), orig_w - 1))
            abs_y = max(0, min(int(coordinates[1] * sy), orig_h - 1))
            match = LocateMatch(
                label=description, x=abs_x, y=abs_y,
                box=None, confidence=0.7,
            )
            return [match], elapsed_ms
        else:
            log.warning("Unexpected box format: %s", raw_output)
            return None

        x1 = max(0, min(int(rx1 * sx), orig_w - 1))
        y1 = max(0, min(int(ry1 * sy), orig_h - 1))
        x2 = max(0, min(int(rx2 * sx), orig_w - 1))
        y2 = max(0, min(int(ry2 * sy), orig_h - 1))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        match = LocateMatch(
            label=description, x=cx, y=cy,
            box=(x1, y1, x2, y2), confidence=0.85,
        )
        return [match], elapsed_ms

    def _locate_with_zonui(
        self, image_b64: str, description: str, mode: str = "point",
    ) -> tuple[list[LocateMatch], float] | None:
        """Use ZonUI-3B to directly locate an element by description.

        For mode='point': returns a single click point.
        For mode='box': asks for top-left and bottom-right coordinates.
        Returns None if ZonUI-3B is not loaded or fails.
        """
        if not self._zonui_loaded:
            return None

        import torch
        from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

        t0 = time.perf_counter()
        img = _decode_image(image_b64)
        orig_w, orig_h = img.size

        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28

        resized_height, resized_width = smart_resize(
            img.height, img.width,
            factor=self._zonui_processor.image_processor.patch_size * self._zonui_processor.image_processor.merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        resized_image = img.resize((resized_width, resized_height))

        if mode == "box":
            _SYSTEM = (
                "Based on the screenshot of the page, I give a text description and you "
                "give the bounding box of the described element. Return the top-left and "
                "bottom-right coordinates as [[x1, y1], [x2, y2]]."
            )
        else:
            _SYSTEM = (
                "Based on the screenshot of the page, I give a text description and you "
                "give its corresponding location. The coordinate represents a clickable "
                "location [x, y] for an element."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _SYSTEM},
                    {"type": "image", "image": resized_image, "min_pixels": min_pixels, "max_pixels": max_pixels},
                    {"type": "text", "text": description},
                ],
            }
        ]

        try:
            text = self._zonui_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self._zonui_processor(
                text=[text], images=[resized_image],
                return_tensors="pt", training=False,
            ).to(self._zonui_model.device)

            with torch.inference_mode():
                generated_ids = self._zonui_model.generate(**inputs)

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._zonui_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            log.info("ZonUI-3B raw output for %r (mode=%s): %s", description, mode, output_text)

            coordinates = ast.literal_eval(output_text)

            elapsed_ms = (time.perf_counter() - t0) * 1000

            if mode == "box":
                # Expect [[x1,y1],[x2,y2]] or a flat [x1,y1,x2,y2]
                if (isinstance(coordinates, list) and len(coordinates) == 2
                        and isinstance(coordinates[0], list)):
                    # [[x1,y1],[x2,y2]]
                    rx1, ry1 = coordinates[0]
                    rx2, ry2 = coordinates[1]
                elif isinstance(coordinates, list) and len(coordinates) == 4:
                    rx1, ry1, rx2, ry2 = coordinates
                elif isinstance(coordinates, list) and len(coordinates) == 2:
                    # ZonUI only returned a point — use it as center with no box
                    log.info("ZonUI-3B returned point in box mode, using as center")
                    norm_x = coordinates[0] / resized_width
                    norm_y = coordinates[1] / resized_height
                    abs_x = int(norm_x * orig_w)
                    abs_y = int(norm_y * orig_h)
                    abs_x = max(0, min(abs_x, orig_w - 1))
                    abs_y = max(0, min(abs_y, orig_h - 1))
                    match = LocateMatch(
                        label=description, x=abs_x, y=abs_y,
                        box=None, confidence=0.7,
                    )
                    return [match], elapsed_ms
                else:
                    log.warning("ZonUI-3B returned unexpected box format: %s", output_text)
                    return None

                # Convert from resized image space to original
                x1 = int((rx1 / resized_width) * orig_w)
                y1 = int((ry1 / resized_height) * orig_h)
                x2 = int((rx2 / resized_width) * orig_w)
                y2 = int((ry2 / resized_height) * orig_h)

                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                match = LocateMatch(
                    label=description, x=cx, y=cy,
                    box=(x1, y1, x2, y2), confidence=0.85,
                )
                return [match], elapsed_ms
            else:
                # Point mode
                if len(coordinates) != 2:
                    log.warning("ZonUI-3B returned unexpected format: %s", output_text)
                    return None

                norm_x = coordinates[0] / resized_width
                norm_y = coordinates[1] / resized_height
                abs_x = int(norm_x * orig_w)
                abs_y = int(norm_y * orig_h)

                abs_x = max(0, min(abs_x, orig_w - 1))
                abs_y = max(0, min(abs_y, orig_h - 1))

                match = LocateMatch(
                    label=description, x=abs_x, y=abs_y,
                    box=None, confidence=0.9,
                )
                return [match], elapsed_ms

        except Exception as exc:
            log.warning("ZonUI-3B locate failed for %r: %s", description, exc)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return None

    def _merge_label(self, caption: str, ocr_text: str) -> str:
        """Combine caption and OCR text into a single label.

        Prefers short, clean OCR text (likely a button label or menu item)
        over verbose captions. Falls back to caption if OCR is empty or
        looks like noise.
        """
        ocr_clean = ocr_text.strip().strip(".")
        if not ocr_clean:
            return caption

        # If OCR returned something short and crisp (like "Easy", "Submit",
        # "Cancel"), prefer it over a caption like "a white rounded button."
        if len(ocr_clean) <= 40 and not ocr_clean.startswith(("<", "{")):
            return ocr_clean

        return caption

    def locate(
        self,
        image_b64: str,
        description: str,
        mode: str = "point",
        confidence_threshold: float = 0.3,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> tuple[list[LocateMatch], float]:
        if mode == "box":
            return self._locate_box(image_b64, description, confidence_threshold,
                                    target_width, target_height)

        # Point/all mode: KV-Ground first, ZonUI second, OmniParser fallback
        kvground_result = self._locate_with_kvground(
            image_b64, description, mode="point",
            target_width=target_width, target_height=target_height,
        )
        if kvground_result is not None:
            matches, elapsed_ms = kvground_result
            if matches:
                return matches, elapsed_ms

        zonui_result = self._locate_with_zonui(image_b64, description, mode="point")
        if zonui_result is not None:
            matches, elapsed_ms = zonui_result
            if matches:
                return matches, elapsed_ms

        # Fallback: OmniParser detect-then-match approach
        return self._locate_with_omniparser(image_b64, description, confidence_threshold)

    def _locate_box(
        self,
        image_b64: str,
        description: str,
        confidence_threshold: float = 0.3,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> tuple[list[LocateMatch], float]:
        """Box mode: KV-Ground first, then OmniParser, then ZonUI."""
        # Primary: KV-Ground-4B
        kvground_result = self._locate_with_kvground(
            image_b64, description, mode="box",
            target_width=target_width, target_height=target_height,
        )
        if kvground_result is not None:
            matches, elapsed_ms = kvground_result
            if matches:
                return matches, elapsed_ms

        # Secondary: OmniParser which has native bounding boxes
        if self._loaded:
            matches, elapsed_ms = self._locate_with_omniparser(
                image_b64, description, confidence_threshold, prefer_largest=True,
            )
            if matches:
                return matches, elapsed_ms

        # Tertiary: ZonUI asked for bounding box coordinates
        zonui_result = self._locate_with_zonui(image_b64, description, mode="box")
        if zonui_result is not None:
            matches, elapsed_ms = zonui_result
            if matches:
                return matches, elapsed_ms

        return [], 0.0

    def _locate_with_omniparser(
        self,
        image_b64: str,
        description: str,
        confidence_threshold: float = 0.3,
        prefer_largest: bool = False,
    ) -> tuple[list[LocateMatch], float]:
        """OmniParser detect-then-match approach with bounding boxes."""
        elements, parse_ms = self.parse(image_b64)

        t0 = time.perf_counter()
        desc_lower = description.lower().strip()
        scored: list[tuple[float, UIElement]] = []

        for elem in elements:
            candidates = [elem.label]
            if elem.ocr_text:
                candidates.append(elem.ocr_text)
            if elem.caption and elem.caption != elem.label:
                candidates.append(elem.caption)

            best_ratio = 0.0
            for candidate in candidates:
                cand_lower = candidate.lower().strip()
                ratio = SequenceMatcher(None, desc_lower, cand_lower).ratio()
                if desc_lower in cand_lower or cand_lower in desc_lower:
                    ratio = max(ratio, 0.8)
                if desc_lower == cand_lower:
                    ratio = 1.0
                desc_words = set(desc_lower.split())
                cand_words = set(cand_lower.split())
                if desc_words & cand_words and len(desc_words & cand_words) / len(desc_words) > 0.5:
                    ratio = max(ratio, 0.6)
                best_ratio = max(best_ratio, ratio)

            scored.append((best_ratio, elem))

        # Sort by score; for box mode, break ties by preferring larger elements
        if prefer_largest:
            scored.sort(key=lambda t: (t[0], t[1].width * t[1].height), reverse=True)
        else:
            scored.sort(key=lambda t: t[0], reverse=True)

        matches: list[LocateMatch] = []
        for score, elem in scored:
            if score < confidence_threshold:
                continue
            cx, cy = elem.center
            matches.append(LocateMatch(
                label=elem.label, x=cx, y=cy,
                box=(elem.x1, elem.y1, elem.x2, elem.y2),
                confidence=score,
            ))

        match_ms = (time.perf_counter() - t0) * 1000
        return matches, parse_ms + match_ms
