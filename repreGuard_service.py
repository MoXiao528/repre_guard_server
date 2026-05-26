from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

import torch

from config import settings
from openai_roberta_detector import OpenAIRobertaDetector

LOGGER = logging.getLogger(__name__)

MODEL_NAME = settings.model_name
MODEL_CACHE_DIR = settings.model_cache_dir
MAX_INPUT_TOKENS = settings.max_input_tokens

_detector: OpenAIRobertaDetector | None = None


class DetectServiceError(Exception):
    status_code = 500
    code = "DETECT_INTERNAL_ERROR"

    def __init__(self, message: str, *, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class InputTooLongError(DetectServiceError):
    status_code = 422
    code = "INPUT_TOO_LONG"

    def __init__(self, current_tokens: int, max_tokens: int) -> None:
        super().__init__(
            f"Input exceeds the model limit of {max_tokens} tokens.",
            detail={"current_tokens": current_tokens, "max_tokens": max_tokens},
        )


class CudaOOMError(DetectServiceError):
    status_code = 503
    code = "CUDA_OOM"

    def __init__(self) -> None:
        super().__init__("CUDA ran out of memory while processing this request.")


def _resolve_device() -> str:
    if settings.device in {"cpu", "cuda"}:
        if settings.device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA was requested but is unavailable, detector will fall back to CPU.")
            return "cpu"
        return settings.device

    if settings.device != "auto":
        LOGGER.warning("Unsupported REPRE_GUARD_DEVICE=%s, detector will use auto mode.", settings.device)

    if torch.cuda.is_available():
        return "cuda"
    LOGGER.warning("CUDA is unavailable, detector will fall back to CPU.")
    return "cpu"


def _cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_model_source() -> str:
    return settings.resolve_model_source()


def _init_detector() -> OpenAIRobertaDetector:
    device = _resolve_device()
    model_source = _resolve_model_source()
    return OpenAIRobertaDetector(
        model_name_or_path=model_source,
        cache_dir=str(MODEL_CACHE_DIR),
        device=device,
        max_length=MAX_INPUT_TOKENS,
        revision=settings.model_revision,
        local_files_only=settings.local_files_only,
        threshold=settings.threshold,
        ai_label_id=settings.ai_label_id,
        tokenizer_use_fast=settings.tokenizer_use_fast,
    )


def get_detector() -> OpenAIRobertaDetector:
    global _detector
    if _detector is None:
        LOGGER.info("Initializing RepreGuard detector pipeline, model=%s", MODEL_NAME)
        _detector = _init_detector()
    return _detector


def _count_input_tokens(text: str) -> int:
    detector = get_detector()
    token_count = detector.count_tokens(text)
    if token_count > MAX_INPUT_TOKENS:
        raise InputTooLongError(token_count, MAX_INPUT_TOKENS)
    return token_count


def compute_repre_score(text: str) -> float:
    detector = get_detector()
    return detector.score_text(text)


@dataclass
class DetectResult:
    text: str
    score: float
    threshold: float
    label: str
    model: str
    score_type: str


def detect_text(text: str) -> DetectResult:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise DetectServiceError("Text cannot be empty.", detail={"code": "TEXT_EMPTY"})

    _count_input_tokens(normalized_text)

    try:
        prediction = get_detector().predict_text(normalized_text)
    except torch.cuda.OutOfMemoryError as exc:
        LOGGER.exception("CUDA OOM while scoring text.")
        _cleanup_cuda_memory()
        raise CudaOOMError() from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            LOGGER.exception("RuntimeError matched CUDA OOM while scoring text.")
            _cleanup_cuda_memory()
            raise CudaOOMError() from exc
        raise

    return DetectResult(
        text=normalized_text,
        score=float(prediction.raw_score),
        threshold=float(prediction.threshold),
        label=prediction.label,
        model=MODEL_NAME,
        score_type=prediction.score_type,
    )
