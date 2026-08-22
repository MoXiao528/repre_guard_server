import asyncio
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from config import settings
from ingress import DetectorIngressMiddleware
from repreGuard_service import DetectResult, DetectServiceError, detect_text, get_detector


class DetectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class DetectResponse(BaseModel):
    score: float
    threshold: float
    label: Literal["AI", "HUMAN"]
    model_name: str
    score_type: Literal["probability"] = "probability"


app = FastAPI(title="RepreGuard Detect Service")
app.add_middleware(DetectorIngressMiddleware, service_token=settings.service_token)
GPU_SEMAPHORE = asyncio.Semaphore(1)


@app.on_event("startup")
def load_detector() -> None:
    """Initialize the detector pipeline once at process startup."""
    settings.require_service_token()
    get_detector()


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest) -> DetectResponse:
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "TEXT_EMPTY", "message": "Text cannot be empty."},
        )

    try:
        async with GPU_SEMAPHORE:
            result: DetectResult = await run_in_threadpool(detect_text, text)
    except DetectServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "DETECT_INTERNAL_ERROR",
                "message": "Detect service failed unexpectedly.",
            },
        ) from exc

    return DetectResponse(
        score=result.score,
        threshold=result.threshold,
        label=result.label,
        model_name=result.model,
        score_type=result.score_type,
    )
