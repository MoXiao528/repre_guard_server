import json
import logging
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from repreGuard_detector import AIHumanFunctionModel

LOGGER = logging.getLogger(__name__)

# 将下方的 "sshleifer/tiny-gpt2" 替换为MODEL_NAME = "Qwen/Qwen2.5-7B"
MODEL_NAME = "sshleifer/tiny-gpt2"

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

    # 2. 数据截断
    # 4090 显存较大，且 Qwen 支持长文本。
    # 建议改为 2048，这样能保留更多语义信息，提升 RepReader 的准确性。
    # 注意：如果你发现显存溢出 (OOM)，可以将这里回调至 1024。
    train_data = truncate_data(train_data, MODEL_NAME, max_length=450)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")

    model = AIHumanFunctionModel(
        model_name_or_path=MODEL_NAME,
        ntrain=len(train_data),
        rep_token=-1,

        # 当前 batch_size=16 仅适用于 tiny 模型。
        # 换成 7B 模型后，必须将其改为 1 或 2！
        # 如果不改，必定报错 CUDA Out of Memory。
        # 7B 模型在 4090 (24GB) 上运行，必须设为 1 或 2。
        # 设为 16 必死无疑 (OOM)。
        # 建议先用 1 跑通，如果显存还有剩（可以用 nvidia-smi 观察），再尝试改为 2。
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