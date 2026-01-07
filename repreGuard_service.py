from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from repreGuard_detector import AIHumanFunctionModel

LOGGER = logging.getLogger(__name__)

MODEL_NAME = "sshleifer/tiny-gpt2"
#MODEL_NAME = "Qwen/Qwen2.5-7B"

THRESHOLD = 2.4924452377944597

READER_PATH = Path("saved_rep_reader.pt")

_detector: AIHumanFunctionModel | None = None


def _ensure_cuda_available() -> str:
    if torch.cuda.is_available():
        return "cuda"
    LOGGER.warning("CUDA 不可用，已降级到 CPU。")
    return "cpu"


def _load_rep_reader(path: Path):
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _init_detector() -> AIHumanFunctionModel:
    device = _ensure_cuda_available()
    model = AIHumanFunctionModel(
        model_name_or_path=MODEL_NAME,
        ntrain=128,
        rep_token=-1,
        batch_size=16,
        random_seed=2025,
        device=device,
    )
    model.rep_reader = _load_rep_reader(READER_PATH)
    if model.rep_reader is None:
        raise RuntimeError(
            f"未找到 rep_reader 文件: {READER_PATH}. "
            "请先运行 init_tiny_model.py 生成 saved_rep_reader.pt。"
        )
    return model


def get_detector() -> AIHumanFunctionModel:
    global _detector
    if _detector is None:
        LOGGER.info("正在初始化 RepreGuard 检测链路，模型=%s", MODEL_NAME)
        _detector = _init_detector()
    return _detector


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


def detect_text(text: str) -> DetectResult:
    score = compute_repre_score(text)
    label = "AI" if score > THRESHOLD else "HUMAN"
    return DetectResult(
        text=text,
        score=float(score),
        threshold=THRESHOLD,
        label=label,
        model=MODEL_NAME,
    )
