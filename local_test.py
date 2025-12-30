# local_test.py
"""
这是一个用于本地测试的最小检测逻辑：
- 使用 tiny 模型 sshleifer/tiny-gpt2
- 做一个简单的 "score"：取最后一层 hidden state 的 L2 范数均值
- 用外部给的阈值做分类
"""

from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


TINY_MODEL_NAME = "sshleifer/tiny-gpt2"
THRESHOLD = 2.4924452377944597  # 先占位用同一阈值，真实部署会换成Qwen下的阈值


# 为了避免每次调用都重新加载模型，用模块级全局变量缓存
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"[INFO] Loading model: {TINY_MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(TINY_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def compute_repre_score(text: str) -> float:
    """
    临时简化版的“RepreScore”：
    - 用 tiny-gpt2 计算 hidden_states
    - 取最后一层 hidden_state 的 L2 范数后做平均，作为一个标量 score

    真实 RepreGuard 的实现会复杂得多（特征方向、校准等），
    但这里的目的只是验证“模型能跑 + 服务能通”。
    """
    tokenizer, model = _load_model()

    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        outputs = model(**inputs, output_hidden_states=True)
        # hidden_states 是一个 tuple：每层 [batch, seq_len, hidden_size]
        hidden_states: List[torch.Tensor] = outputs.hidden_states
        last_hidden = hidden_states[-1]  # 取最后一层

        # 一个很简单的“score”：对 batch 和 seq_len 求 L2 范数的均值
        # shape: [batch, seq_len, hidden_size] -> 先算每个 token 的 L2，再对所有token取平均
        token_norms = torch.norm(last_hidden, dim=-1)  # [batch, seq_len]
        score_tensor = token_norms.mean()
        score = score_tensor.item()
        return float(score)


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
        model=TINY_MODEL_NAME,
    )


if __name__ == "__main__":
    human_text = "Today I went to the library and studied for my exam. It was a bit boring, but also satisfying."
    ai_text = "In a rapidly evolving digital landscape, artificial intelligence has become a transformative force."

    for t in [human_text, ai_text]:
        result = detect_text(t)
        print("=" * 80)
        print(f"TEXT: {t}")
        print(f"MODEL: {result.model}")
        print(f"SCORE: {result.score:.6f}")
        print(f"THRESHOLD: {result.threshold:.6f}")
        print(f"LABEL: {result.label}")
