from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List

import torch

from repreGuard_detector import AIHumanFunctionModel

LOGGER = logging.getLogger(__name__)

# TODO: 以后替换检测模型时，只需要修改这里的 MODEL_NAME 即可。
# 例如改成 "meta-llama/Llama-3.1-8B" 或其他你想要的模型名称/路径。
MODEL_NAME = "sshleifer/tiny-gpt2"

# TODO: 以后替换检测模型时，一并更新阈值。
THRESHOLD = 2.4924452377944597

# TODO: 训练数据用于拟合 RepreGuard 的方向向量，请替换成你自己的训练集路径。
# 目前仅做推理，不再读取训练数据；保留读取逻辑以便未来需要时一键恢复。
DEFAULT_TRAIN_DATA_PATH = "direct_prompt_train.json"
TRAIN_DATA_PATH = os.environ.get("REPRE_GUARD_TRAIN_DATA", DEFAULT_TRAIN_DATA_PATH)
NTRAIN = int(os.environ.get("REPRE_GUARD_NTRAIN", "0"))
FIT_ON_STARTUP = os.environ.get("REPRE_GUARD_FIT_ON_STARTUP", "1") == "1"

# TODO: 推理模式下从已保存的 rep_reader 读取方向向量（避免再训练）。
# 以后替换模型/方向向量时，只需更新这个路径即可。
READER_PATH = os.environ.get("REPRE_GUARD_READER_PATH", "")

_detector: AIHumanFunctionModel | None = None


def _ensure_cuda_available() -> str:
    if torch.cuda.is_available():
        return "cuda"
    LOGGER.warning("CUDA 不可用，已降级到 CPU。")
    return "cpu"


def _load_train_data(path: str, ntrain: int) -> List[dict]:
    if not path:
        raise RuntimeError("未设置 REPRE_GUARD_TRAIN_DATA，无法初始化 RepreGuard 检测链路。")
    LOGGER.info("准备加载训练集: %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"训练集不存在: {path}，请确认已将 {DEFAULT_TRAIN_DATA_PATH} 放在项目根目录，"
            "或设置 REPRE_GUARD_TRAIN_DATA 指向训练集。"
        )
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    LOGGER.info("训练集加载完成，总样本数=%s", len(data))
    if not isinstance(data, list):
        raise ValueError(f"训练集数据格式错误，期望 list，实际为 {type(data)}。")
    if not data:
        raise ValueError("训练集数据为空，无法拟合方向向量。")
    if ntrain and ntrain > 0:
        return data[:ntrain]
    return data


def _load_rep_reader(path: str):
    if not path:
        return None
    return torch.load(path, map_location="cpu")


def _init_detector() -> AIHumanFunctionModel:
    device = _ensure_cuda_available()
    model = AIHumanFunctionModel(
        model_name_or_path=MODEL_NAME,
        ntrain=NTRAIN,
        rep_token=-1,
        batch_size=16,
        random_seed=2025,
        device=device,
    )
    model.rep_reader = _load_rep_reader(READER_PATH)
    if model.rep_reader is None:
        if FIT_ON_STARTUP:
            LOGGER.info("未提供 REPRE_GUARD_READER_PATH，将尝试使用训练集拟合方向向量。")
            train_data = _load_train_data(TRAIN_DATA_PATH, NTRAIN)
            LOGGER.info("开始拟合方向向量，训练样本数=%s", len(train_data))
            model.fit_rep_reader(train_data)
            LOGGER.info("方向向量拟合完成。")
        else:
            LOGGER.warning(
                "未设置 REPRE_GUARD_READER_PATH，且 REPRE_GUARD_FIT_ON_STARTUP=0，"
                "当前不会读取训练数据并拟合方向向量。"
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
