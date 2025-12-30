import os
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

os.environ.setdefault("HF_HOME", "D:/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/huggingface/cache")

# 这里复用你已经验证过的 local_test.py
from local_test import detect_text, DetectResult


class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
    score: float
    threshold: float
    label: Literal["AI", "HUMAN"]
    model_name: str


app = FastAPI(title="RepreGuard Detect Service (tiny model)")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    """
    调用 local_test.detect_text，对文本进行检测。
    """
    # 这里 FastAPI 会自动把 JSON 解析成 DetectRequest
    result: DetectResult = detect_text(req.text)

    return DetectResponse(
        score=result.score,
        threshold=result.threshold,
        label=result.label,
        model_name=result.model,
    )
