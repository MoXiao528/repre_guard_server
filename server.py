from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

# 使用原有 RepreGuard 检测链路
from repreGuard_service import detect_text, DetectResult, get_detector


class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
    score: float
    threshold: float
    label: Literal["AI", "HUMAN"]
    model_name: str


app = FastAPI(title="RepreGuard Detect Service (tiny model)")


@app.on_event("startup")
def load_detector() -> None:
    """
    启动时初始化 RepreGuard 检测链路。
    """
    get_detector()


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    """
    调用 RepreGuard 检测链路，对文本进行检测。
    """
    # 这里 FastAPI 会自动把 JSON 解析成 DetectRequest
    result: DetectResult = detect_text(req.text)

    return DetectResponse(
        score=result.score,
        threshold=result.threshold,
        label=result.label,
        model_name=result.model,
    )
