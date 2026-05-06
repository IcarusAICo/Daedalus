"""FastAPI grounding microservice for Daedalus.

Endpoints:
  POST /parse    -- parse all UI elements from a screenshot
  POST /locate   -- find elements matching a natural-language description
  GET  /health   -- health check
"""

from __future__ import annotations

import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException

from engine import GroundingEngine
from models import (
    LocateRequest,
    LocateResponse,
    ParseRequest,
    ParseResponse,
    UIElement,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Daedalus Grounding Service",
    description="Visual grounding microservice for UI element detection and location.",
    version="0.1.0",
)

engine = GroundingEngine()


@app.on_event("startup")
async def startup() -> None:
    device = os.environ.get("GROUNDING_DEVICE", "cuda")
    log.info("loading grounding models on %s...", device)
    engine.load(device=device)
    if engine.is_loaded:
        log.info("grounding engine ready")
    else:
        log.warning("grounding engine running in stub mode (no models loaded)")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": engine.is_loaded,
        "zonui_loaded": engine._zonui_loaded,
        "omniparser_loaded": engine._loaded,
    }


@app.post("/parse", response_model=ParseResponse)
async def parse(req: ParseRequest) -> ParseResponse:
    """Parse all UI elements from a screenshot."""
    try:
        elements, elapsed_ms = engine.parse(req.image_b64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parse failed: {exc}") from exc
    return ParseResponse(elements=elements, parse_time_ms=round(elapsed_ms, 2))


@app.post("/locate", response_model=LocateResponse)
async def locate(req: LocateRequest) -> LocateResponse:
    """Locate elements matching a natural-language description."""
    try:
        matches, elapsed_ms = engine.locate(
            req.image_b64,
            req.description,
            mode=req.mode,
            confidence_threshold=req.confidence_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Locate failed: {exc}") from exc

    return LocateResponse(
        found=len(matches) > 0,
        matches=matches,
        locate_time_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    port = int(os.environ.get("GROUNDING_PORT", "8420"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
