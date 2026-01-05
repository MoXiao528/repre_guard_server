import json
import logging
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from repreGuard_detector import AIHumanFunctionModel

LOGGER = logging.getLogger(__name__)

# TODO: Update MODEL_NAME to the production model path when training the final detector.
MODEL_NAME = "sshleifer/tiny-gpt2"

# TODO: Point TRAIN_DATA_PATH to the full, real training dataset when ready for production.
TRAIN_DATA_PATH = Path("train_MIXED_ALL.json")

READER_OUTPUT_PATH = Path("saved_rep_reader.pt")


def _load_train_data() -> list[dict]:
    if TRAIN_DATA_PATH.exists():
        LOGGER.info("Loading training data from %s", TRAIN_DATA_PATH)
        with TRAIN_DATA_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    LOGGER.warning("Training data not found. Using synthetic dummy samples.")
    dummy_data = []
    for idx in range(10):
        dummy_data.append(
            {
                "direct_prompt": f"AI synthetic sample {idx}: {random.random()}",
                "human_text": f"Human synthetic sample {idx}: {random.random()}",
            }
        )
    return dummy_data


def truncate_data(data: list[dict], model_name: str, max_length: int = 400) -> list[dict]:
    """
    使用 tokenizer 对数据中的文本字段进行截断。
    注意：GPT-2 上下文限制为 1024。
    如果 pipeline 将 'prompt' 和 'text' 拼接，那么单个字段必须远小于 1024。
    这里我们将单个字段限制在 400 左右，确保 400+400 < 1024。
    """
    LOGGER.info(f"Initializing tokenizer for truncation (max_length={max_length})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        LOGGER.warning(f"Failed to load tokenizer for truncation: {e}. Skipping truncation.")
        return data

    truncated_count = 0
    # 遍历所有数据
    for item in data:
        for key, value in item.items():
            if isinstance(value, str):
                # 编码
                tokens = tokenizer.encode(value, add_special_tokens=False)
                # 检查长度
                if len(tokens) > max_length:
                    # 截断
                    item[key] = tokenizer.decode(tokens[:max_length])
                    truncated_count += 1

    if truncated_count > 0:
        LOGGER.info(f"Truncated {truncated_count} fields to {max_length} tokens to prevent overflow.")

    return data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 1. 加载数据
    train_data = _load_train_data()

    # 2. 激进截断 (Aggressive Truncation)
    # 设置为 450，假设最坏情况是两个字段拼接：450 + 450 = 900 < 1024
    # 这样留出了 124 个 token 给特殊符号或其他开销，非常安全。
    train_data = truncate_data(train_data, MODEL_NAME, max_length=450)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")

    model = AIHumanFunctionModel(
        model_name_or_path=MODEL_NAME,
        ntrain=len(train_data),
        rep_token=-1,
        batch_size=16,
        random_seed=2025,
        device=device,
    )

    LOGGER.info("Fitting rep_reader directions with %d samples.", len(train_data))

    try:
        model.fit_rep_reader(train_data)
    except RuntimeError as e:
        if "device-side assert" in str(e):
            LOGGER.error("!!! CUDA ERROR DETECTED !!!")
            LOGGER.error("Please RESTART your Python kernel/terminal explicitly.")
            LOGGER.error("The GPU state is corrupted from a previous error and cannot recover without a restart.")
        raise e

    READER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.rep_reader, READER_OUTPUT_PATH)
    LOGGER.info("Saved rep_reader to %s", READER_OUTPUT_PATH)


if __name__ == "__main__":
    main()