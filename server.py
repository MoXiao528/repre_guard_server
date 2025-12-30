import os
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

os.environ.setdefault("HF_HOME", "D:/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/huggingface/cache")

# 这里复用你已经验证过的 local_test.py
from repreGuard_detector import RepreGuardDetector, DetectResult, DEFAULT_MODEL_NAME, DEFAULT_THRESHOLD


class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
    score: float
    threshold: float
    label: Literal["AI", "HUMAN"]
    model_name: str


app = FastAPI(title="RepreGuard Detect Service (tiny model)")


DEFAULT_TRAIN_DATA_PATH = "datasets/detectrl_dataset/main_dataset/detectrl_train_dataset_llm_type_ChatGPT.json"
_detector: RepreGuardDetector | None = None


def _get_detector() -> RepreGuardDetector:
    global _detector
    if _detector is None:
        train_data_path = os.getenv("REPRE_GUARD_TRAIN_DATA_PATH", DEFAULT_TRAIN_DATA_PATH)
        ntrain = int(os.getenv("REPRE_GUARD_NTRAIN", "128"))
        rep_token = float(os.getenv("REPRE_GUARD_REP_TOKEN", "-1"))
        batch_size = int(os.getenv("REPRE_GUARD_BATCH_SIZE", "16"))
        random_seed = int(os.getenv("REPRE_GUARD_RANDOM_SEED", "2025"))
        model_name = os.getenv("REPRE_GUARD_MODEL_NAME", DEFAULT_MODEL_NAME)
        threshold = float(os.getenv("REPRE_GUARD_THRESHOLD", str(DEFAULT_THRESHOLD)))
        _detector = RepreGuardDetector(
            train_data_path=train_data_path,
            model_name_or_path=model_name,
            threshold=threshold,
            ntrain=ntrain,
            rep_token=rep_token,
            batch_size=batch_size,
            random_seed=random_seed,
        )
    return _detector


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest) -> DetectResponse:
    """
    调用 RepreGuard 的检测流程，对文本进行检测。
    """
    # 这里 FastAPI 会自动把 JSON 解析成 DetectRequest
    result: DetectResult = _get_detector().detect_text(req.text)

    return DetectResponse(
        score=result.score,
        threshold=result.threshold,
        label=result.label,
        model_name=result.model,
    )
